;; cin_export.il v2 — hardened Virtuoso -> CIN exporter (M2)
;;
;; v2 changes vs the M0 prototype:
;;   - device mapping driven by CDF simInfo (spectre section): componentName
;;     identifies analogLib primitives; termOrder gives terminal order
;;   - transistor classification by terminal-name heuristics (d/g/s/b -> mosfet,
;;     c/b/e -> bjt), polarity from the cell name, with a manual override table
;;   - hierarchy descent: a cell with no device mapping but a schematic view
;;     becomes a CIN subckt definition (recursively)
;;   - global nets (trailing '!') collected automatically
;;   - w/l/nf/m captured into params/meta
;;   - loud reports: skipped cells, unknown cells, unsupported constructs
;;
;; Known limitations (reported when hit): no Hierarchy-Editor config views
;; (plain schematic descent only), no buses / iterated instances.
;;
;; Usage in the CIW (dir from `circuitinsight-skill-path` in a shell):
;;   load("<skill-dir>/cin_export.il")
;;   CInExport("worklib" "tb_m0" "schematic"
;;             "CircuitInsight/m0/tb_m0.cin.json" ?ground list("0" "gnd!"))

;; ---------------------------------------------------------------- config

;; net-tie / annotation symbols: expected non-devices, skipped silently
(setq CInIgnoreCells (list "gnd" "vdd" "vcc" "vss" "noConn" "patchcord"
                           "ipin" "opin" "iopin" "sympin" "cds_thru"))

;; spectre simInfo componentName -> CIN device_type
(setq CInComponentMap
  (list (list "resistor"  "resistor")
        (list "capacitor" "capacitor")
        (list "inductor"  "inductor")
        (list "vsource"   "vsource")
        (list "isource"   "isource")
        (list "vcvs"      "vcvs")
        (list "vccs"      "vccs")
        (list "cccs"      "cccs")
        (list "ccvs"      "ccvs")
        (list "diode"     "diode")
        (list "balun"     "balun")     ; analogLib ideal_balun
        (list "relay"     "switch")    ; analogLib sp1tswitch et al.
        (list "switch"    "switch")))

;; manual overrides for cells the heuristics misclassify:
;; entries: list(cellName device_type polarity_or_nil)
;; Defaults cover analogLib testbench macros that would otherwise be descended
;; into or misread (ideal_balun netlists as a transformer subckt; sp1tswitch is
;; the 2-terminal spectre ideal switch).
(setq CInDeviceOverrides
  (list (list "ideal_balun" "balun"  nil)
        (list "sp1tswitch"  "switch" nil)))

;; ---------------------------------------------------------------- helpers

(procedure (CInJsonStr s)
  (strcat "\""
          (pcreReplace (pcreCompile "([\\\\\"])") (sprintf nil "%s" (or s ""))
                       "\\\\\\1" 0)
          "\""))

;; nil-safe stringification (SKILL sprintf errors on %s with nil, and %s is
;; only safe for strings/symbols; %L handles numbers)
(procedure (CInStr x)
  (cond ((null x) "")
        ((stringp x) x)
        ((symbolp x) (sprintf nil "%s" x))
        (t (sprintf nil "%L" x))))

(procedure (CInLower s) (lowerCase (CInStr s)))

(procedure (CInSimInfo libName cellName)
  (let (cdfId)
    (setq cdfId (car (errset (cdfGetCellCDF (ddGetObj libName cellName)))))
    (when cdfId (car (errset (getq (getq cdfId simInfo) spectre))))))

(procedure (CInTermOrder sim)
  ;; termOrder entries may be symbols or strings; normalize to strings
  (let ((to (when sim (car (errset (getq sim termOrder))))))
    (when (listp to)
      (foreach mapcar x to (CInStr x)))))

(procedure (CInMatchAny name patterns)
  (exists p patterns (rexMatchp p name)))

;; classify a master cell: returns
;;   list("device" device_type polarity termOrderStrings)
;; | list("subckt" schematicCellView)
;; | list("ignore") | list("unknown")
(procedure (CInClassify libName cellName)
  (let (sim comp terms lterms ov dt pol lname)
    (setq lname (CInLower cellName))
    (cond
      ((member cellName CInIgnoreCells) (list "ignore"))
      (t
       (setq sim (CInSimInfo libName cellName))
       (setq comp
         (when sim
           (let ((c (car (errset (getq sim componentName)))))
             (when c (CInStr c)))))
       (setq terms (CInTermOrder sim))
       (setq lterms (foreach mapcar x (or terms nil) (CInLower x)))
       (setq ov (car (exists e CInDeviceOverrides (equal (car e) cellName))))
       (cond
         (ov (list "device" (cadr ov) (caddr ov) terms))
         ;; analogLib-style spectre primitive
         ((and comp (setq dt (cadr (car (exists e CInComponentMap
                                           (equal (car e) comp))))))
          (list "device" dt nil terms))
         ;; MOS heuristic: exactly d g s b
         ((and terms (equal (length terms) 4)
               (member "d" lterms) (member "g" lterms)
               (member "s" lterms) (member "b" lterms))
          (setq pol (if (CInMatchAny lname (list "pmos" "pch" "pfet")) "p" "n"))
          (list "device" "mosfet" pol terms))
         ;; BJT heuristic: c b e (optional 4th substrate terminal, dropped)
         ((and terms (member "c" lterms) (member "b" lterms) (member "e" lterms)
               (leqp (length terms) 4))
          (setq pol (if (rexMatchp "pnp" lname) "pnp" "npn"))
          (list "device" "bjt" pol terms))
         ;; hierarchical: descend into the schematic
         ((ddGetObj libName cellName "schematic") (list "subckt"))
         (t (list "unknown")))))))

;; canonical terminal map for a device: list of list(cdfTermName canonicalName)
(procedure (CInTermAssoc dt terms instTermNames)
  (let ((names (or terms instTermNames)) lnames)
    (setq lnames (foreach mapcar x names (CInLower x)))
    (cond
      ((equal dt "mosfet")
       (foreach mapcar x names
         (list x (if (member (CInLower x) (list "d" "g" "s" "b"))
                     (CInLower x) nil))))
      ((equal dt "bjt")
       (foreach mapcar x names
         (list x (if (member (CInLower x) (list "c" "b" "e"))
                     (CInLower x) nil))))    ; substrate etc -> nil (dropped)
      ((equal dt "balun")
       (foreach mapcar x names
         (list x (if (member (CInLower x) (list "d" "c" "p" "n"))
                     (CInLower x) nil))))
      ((member dt (list "vcvs" "vccs" "cccs" "ccvs"))
       (let ((canon (list "p" "n" "cp" "cn")) (i 0) out)
         (foreach x names
           (setq out (cons (list x (nth i canon)) out))
           (setq i (add1 i)))
         (reverse out)))
      (t  ; two-terminal: first -> p, second -> n
       (let ((canon (list "p" "n")) (i 0) out)
         (foreach x names
           (setq out (cons (list x (nth i canon)) out))
           (setq i (add1 i)))
         (reverse out))))))

;; instance parameter capture (best effort; values come from OP data anyway).
;; Empty-string CDF defaults are treated as absent.
(procedure (CInInstParam inst pname)
  (let ((v (CInStr (car (errset (get inst pname))))))
    (unless (equal v "") v)))

;; ---------------------------------------------------------------- emission
;; Dynamic-scoped collectors, reset by CInExport:
;;   CInDefsOrder: def names in emission order
;;   CInDefsPorts / CInDefsBody: assoc lists defName -> data
;;   CInSkipped / CInUnknown / CInUnsupported: report lists
;;   CInGlobals: global nets seen; CInGroundNets: ground list

(procedure (CInNoteNet nname)
  (when (and (rexMatchp "!$" nname)
             (null (member nname CInGroundNets))
             (null (member nname CInGlobals)))
    (setq CInGlobals (cons nname CInGlobals)))
  nname)

(procedure (CInInstTermJson inst termAssoc where)
  ;; -> JSON fragment for "terminals", or nil on failure (reported)
  (let ((pairs nil) (ok t))
    (foreach it (getq inst instTerms)
      (let ((tname (CInStr (getq (getq it term) name)))
            (net (getq it net)) nname canon)
        (setq nname (when net (CInStr (getq net name))))
        (cond
          ((null nname)
           (printf "CInExport: WARNING %s terminal %s unconnected\n" where tname)
           (setq ok nil))
          ((rexMatchp "<" nname)
           (setq CInUnsupported (cons (sprintf nil "%s: bus net %s" where nname)
                                      CInUnsupported))
           (setq ok nil))
          (t
           (setq canon (cadr (assoc tname termAssoc)))
           (when canon
             (CInNoteNet nname)
             (setq pairs (cons (sprintf nil "%s: %s"
                                        (CInJsonStr canon) (CInJsonStr nname))
                               pairs)))))))
    (when ok (buildString (reverse pairs) ", "))))

(procedure (CInEmitDevice inst dt pol terms where)
  (let (termAssoc tj (params nil) (meta nil) frag)
    (setq termAssoc
      (CInTermAssoc dt terms
        (foreach mapcar it (getq inst instTerms)
          (CInStr (getq (getq it term) name)))))
    (setq tj (CInInstTermJson inst termAssoc where))
    (when tj
      (when pol
        (setq params (cons (sprintf nil "\"polarity\": %s" (CInJsonStr pol)) params)))
      (let ((m (CInInstParam inst 'm)))
        (when m (setq params (cons (sprintf nil "\"m\": %s" (CInJsonStr m)) params))))
      (when (equal dt "switch")
        ;; CDF property names vary (snake_case in the netlist, often camelCase
        ;; in CDF); try both, emit canonical snake_case once
        (let ((done nil))
          (foreach pp (list (list 'position    "position")
                            (list 'dc_position "dc_position")
                            (list 'dcPosition  "dc_position")
                            (list 'ac_position "ac_position")
                            (list 'acPosition  "ac_position"))
            (let ((v (CInInstParam inst (car pp))) (canon (cadr pp)))
              (when (and v (null (member canon done)))
                (setq done (cons canon done))
                (setq params (cons (sprintf nil "%s: %s"
                                            (CInJsonStr canon)
                                            (CInJsonStr v)) params)))))))
      (when (member dt (list "vsource" "isource"))
        ;; AC excitation config: lets the adapter detect superposed AC runs
        ;; (multiple nonzero acm) and prefer xf results for per-source TFs
        (foreach p (list 'acm 'acp)
          (let ((v (CInInstParam inst p)))
            (when v (setq params (cons (sprintf nil "%s: %s"
                                                (CInJsonStr (CInStr p))
                                                (CInJsonStr v)) params))))))
      (foreach p (list 'w 'l 'nf)
        (let ((v (CInInstParam inst p)))
          (when v (setq meta (cons (sprintf nil "%s: %s"
                                            (CInJsonStr (CInStr p))
                                            (CInJsonStr v)) meta)))))
      (setq frag
        (sprintf nil
          "      { \"name\": %s, \"device_type\": %s, \"terminals\": { %s }%s, \"meta\": { \"lib\": %s, \"cell\": %s%s } }"
          (CInJsonStr (getq inst name)) (CInJsonStr dt) tj
          (if params (sprintf nil ", \"params\": { %s }"
                              (buildString (reverse params) ", ")) "")
          (CInJsonStr (getq inst libName)) (CInJsonStr (getq inst cellName))
          (if meta (strcat ", " (buildString (reverse meta) ", ")) "")))
      frag)))

(procedure (CInEmitSubcktInst inst defName where)
  (let ((pairs nil) (ok t))
    (foreach it (getq inst instTerms)
      (let ((tname (CInStr (getq (getq it term) name)))
            (net (getq it net)) nname)
        (setq nname (when net (CInStr (getq net name))))
        (cond
          ((null nname)
           (printf "CInExport: WARNING %s port %s unconnected\n" where tname)
           (setq ok nil))
          (t (CInNoteNet nname)
             (setq pairs (cons (sprintf nil "%s: %s"
                                        (CInJsonStr tname) (CInJsonStr nname))
                               pairs))))))
    (when ok
      (sprintf nil "      { \"name\": %s, \"subckt\": %s, \"terminals\": { %s } }"
               (CInJsonStr (getq inst name)) (CInJsonStr defName)
               (buildString (reverse pairs) ", ")))))

(procedure (CInEnsureDef libName cellName)
  ;; export cellName's schematic as a CIN definition (once); returns defName
  (let ((defName (CInStr cellName)) cv lines ports)
    (unless (member defName CInDefsOrder)
      (setq CInDefsOrder (append CInDefsOrder (list defName)))  ; reserve
      (setq cv (dbOpenCellViewByType libName cellName "schematic" "" "r"))
      (unless cv (error "CInExport: cannot open %s/%s/schematic" libName cellName))
      (setq ports
        (foreach mapcar tr (getq cv terminals)
          (CInStr (getq tr name))))
      (foreach p ports
        (when (rexMatchp "<" p)
          (setq CInUnsupported
            (cons (sprintf nil "%s: bus port %s" defName p) CInUnsupported))))
      (setq lines (CInEmitInstances cv defName))
      (setq CInDefsPorts (cons (list defName ports) CInDefsPorts))
      (setq CInDefsBody (cons (list defName lines) CInDefsBody)))
    defName))

(procedure (CInEmitInstances cv defName)
  (let ((lines nil))
    (foreach inst (getq cv instances)
      (let ((master (getq inst cellName)) (mlib (getq inst libName))
            (iname (getq inst name)) cls where frag)
        (setq where (sprintf nil "%s/%s" defName iname))
        (cond
          ((rexMatchp "<" iname)
           (setq CInUnsupported
             (cons (sprintf nil "%s: iterated instance" where) CInUnsupported)))
          (t
           (setq cls (CInClassify mlib master))
           (cond
             ((equal (car cls) "ignore") t)
             ((equal (car cls) "device")
              (setq frag (CInEmitDevice inst (cadr cls) (caddr cls)
                                        (cadddr cls) where))
              (when frag (setq lines (cons frag lines))))
             ((equal (car cls) "subckt")
              (let ((childDef (CInEnsureDef mlib master)))
                (setq frag (CInEmitSubcktInst inst childDef where))
                (when frag (setq lines (cons frag lines)))))
             (t
              (when (getq inst instTerms)   ; pins/labels have no instTerms
                (setq CInUnknown
                  (cons (list where master mlib) CInUnknown)))))))))
    (reverse lines)))

;; ---------------------------------------------------------------- driver

(procedure (CInExport libName cellName viewName outPath
                      @key (ground (list "0" "gnd!")))
  (let (out topDef ndev)
    ;; reset collectors (dynamic scope)
    (setq CInDefsOrder nil) (setq CInDefsPorts nil) (setq CInDefsBody nil)
    (setq CInSkipped nil) (setq CInUnknown nil) (setq CInUnsupported nil)
    (setq CInGlobals nil) (setq CInGroundNets ground)

    (setq topDef (CInEnsureDef libName cellName))

    (setq out (outfile outPath "w"))
    (unless out (error "CInExport: cannot write %s" outPath))
    (fprintf out "{\n  \"cin_version\": \"0.1\",\n")
    (fprintf out "  \"design\": { \"name\": %s, \"source\": { \"kind\": \"virtuoso\", \"lib\": %s, \"view\": %s } },\n"
             (CInJsonStr cellName) (CInJsonStr libName) (CInJsonStr viewName))
    (fprintf out "  \"top\": %s,\n" (CInJsonStr topDef))
    (fprintf out "  \"ground\": [%s],\n"
             (buildString (foreach mapcar g ground (CInJsonStr g)) ", "))
    (fprintf out "  \"globals\": [%s],\n"
             (buildString (foreach mapcar g CInGlobals (CInJsonStr g)) ", "))
    (fprintf out "  \"definitions\": {\n")
    (let ((defFrags nil))
      (foreach dn CInDefsOrder
        (let ((ports (cadr (assoc dn CInDefsPorts)))
              (body (cadr (assoc dn CInDefsBody))))
          (setq defFrags
            (cons (sprintf nil "    %s: { \"ports\": [%s], \"instances\": [\n%s\n    ] }"
                           (CInJsonStr dn)
                           (buildString (foreach mapcar p ports (CInJsonStr p)) ", ")
                           (buildString body ",\n"))
                  defFrags))))
      (fprintf out "%s\n" (buildString (reverse defFrags) ",\n")))
    (fprintf out "  }\n}\n")
    (close out)

    (setq ndev 0)
    (foreach e CInDefsBody (setq ndev (plus ndev (length (cadr e)))))
    (printf "CInExport: wrote %d instances in %d definition(s) to %s\n"
            ndev (length CInDefsOrder) outPath)
    (when CInGlobals
      (printf "CInExport: globals: %L\n" CInGlobals))
    (when CInUnknown
      (printf "CInExport: %d UNKNOWN cells (fix CInDeviceOverrides or report):\n"
              (length CInUnknown))
      (foreach u CInUnknown (printf "  %s (cell %s, lib %s)\n"
                                    (car u) (cadr u) (caddr u))))
    (when CInUnsupported
      (printf "CInExport: %d UNSUPPORTED constructs:\n" (length CInUnsupported))
      (foreach u CInUnsupported (printf "  %s\n" u)))
    t))
