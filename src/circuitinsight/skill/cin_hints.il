;; cin_hints.il — harvest schematic LAYOUT HINTS from the open design.
;;
;; The CIN export carries the electrical truth; this sidecar carries the
;; geometry the human already drew: instance origins/orients/bboxes,
;; per-terminal positions, wire polylines with their nets, net labels
;; and ports. CircuitInsight re-renders the schematic from these hints
;; with its own SVG symbol library — layout harvested, never invented.
;;
;;   CInHintsExport("myLib" "myCell" "schematic" "/path/out.hints.json")
;;
;; Everything geometric is wrapped in errset: the IC23.1 lesson is that
;; object attributes differ across versions, and a hint we cannot read
;; is a hint we omit — never an export that dies halfway.

(procedure (CInHPt pt)
  (sprintf nil "[%L, %L]" (float (xCoord pt)) (float (yCoord pt))))

(procedure (CInHBox bb)
  (sprintf nil "[%L, %L, %L, %L]"
           (float (xCoord (car bb))) (float (yCoord (car bb)))
           (float (xCoord (cadr bb))) (float (yCoord (cadr bb)))))

;; terminal positions: each instTerm's terminal pin figure, transformed
;; into the instance's placement. Optional — omitted when unreadable.
(procedure (CInHTerms inst)
  (let ((frags nil))
    (foreach it (getq inst instTerms)
      (errset
        (let ((tname (getq (getq it term) name))
              (pins (getq (getq it term) pins)) fig bb cx cy tp)
          (when (and tname pins)
            (setq fig (getq (car pins) fig))
            (when fig
              (setq bb (getq fig bBox))
              (setq cx (quotient (plus (xCoord (car bb))
                                       (xCoord (cadr bb))) 2.0))
              (setq cy (quotient (plus (yCoord (car bb))
                                       (yCoord (cadr bb))) 2.0))
              (setq tp (dbTransformPoint (list cx cy)
                                         (getq inst transform)))
              (setq frags
                (cons (sprintf nil "%s: %s" (CInJsonStr tname)
                               (CInHPt tp)) frags)))))
        t))
    (if frags
        (sprintf nil "{%s}" (buildString (reverse frags) ", "))
        "{}")))

(procedure (CInHInstances cv)
  (let ((lines nil))
    (foreach inst (getq cv instances)
      (unless (rexMatchp "<" (getq inst name))   ; iterated: CIN skips too
        (errset
          (setq lines
            (cons (sprintf nil
                    "{\"name\": %s, \"lib\": %s, \"cell\": %s, \"xy\": %s, \"orient\": %s, \"bbox\": %s, \"terms\": %s}"
                    (CInJsonStr (getq inst name))
                    (CInJsonStr (getq inst libName))
                    (CInJsonStr (getq inst cellName))
                    (CInHPt (getq inst xy))
                    (CInJsonStr (CInStr (getq inst orient)))
                    (CInHBox (getq inst bBox))
                    (CInHTerms inst))
                  lines))
          t)))
    (reverse lines)))

;; wires: line/path shapes with their point lists; the net name when the
;; schematic carries it on the shape, "" otherwise (the composer can
;; fall back to the nearest label).
(procedure (CInHWires cv)
  (let ((lines nil) ot pts nname)
    (foreach sh (getq cv shapes)
      (setq ot (CInStr (getq sh objType)))
      (when (or (equal ot "line") (equal ot "path"))
        (errset
          (progn
            (setq pts (getq sh points))
            (setq nname "")
            (errset (setq nname (or (getq (getq sh net) name) "")) t)
            (when (and pts (geqp (length pts) 2))
              (setq lines
                (cons (sprintf nil "{\"net\": %s, \"points\": [%s]}"
                               (CInJsonStr nname)
                               (buildString (mapcar 'CInHPt pts) ", "))
                      lines))))
          t)))
    (reverse lines)))

(procedure (CInHLabels cv)
  (let ((lines nil) ot)
    (foreach sh (getq cv shapes)
      (setq ot (CInStr (getq sh objType)))
      (when (equal ot "label")
        (errset
          (setq lines
            (cons (sprintf nil "{\"text\": %s, \"xy\": %s}"
                           (CInJsonStr (CInStr (getq sh theLabel)))
                           (CInHPt (getq sh xy)))
                  lines))
          t)))
    (reverse lines)))

(procedure (CInHPins cv)
  (let ((lines nil))
    (foreach term (getq cv terminals)
      (errset
        (let ((pins (getq term pins)) fig bb)
          (when pins
            (setq fig (getq (car pins) fig))
            (when fig
              (setq bb (getq fig bBox))
              (setq lines
                (cons (sprintf nil "{\"name\": %s, \"xy\": [%L, %L]}"
                        (CInJsonStr (getq term name))
                        (float (quotient (plus (xCoord (car bb))
                                               (xCoord (cadr bb))) 2.0))
                        (float (quotient (plus (yCoord (car bb))
                                               (yCoord (cadr bb))) 2.0)))
                      lines)))))
        t))
    (reverse lines)))

(procedure (CInHEmitDef cv defName out)
  (fprintf out "    %s: {\n" (CInJsonStr defName))
  (errset
    (fprintf out "      \"bbox\": %s,\n" (CInHBox (getq cv bBox)))
    t)
  (fprintf out "      \"instances\": [\n        %s\n      ],\n"
           (buildString (CInHInstances cv) ",\n        "))
  (fprintf out "      \"wires\": [\n        %s\n      ],\n"
           (buildString (CInHWires cv) ",\n        "))
  (fprintf out "      \"labels\": [%s],\n"
           (buildString (CInHLabels cv) ", "))
  (fprintf out "      \"pins\": [%s]\n"
           (buildString (CInHPins cv) ", "))
  (fprintf out "    }"))

;; walk the hierarchy the same way the CIN does: the top cell plus every
;; subcircuit master that has a schematic view, keyed by the SAME
;; definition names so the Python side joins hints to CIN definitions.
(procedure (CInHCollectDefs libName cellName viewName)
  (let ((todo (list (list libName cellName))) (done nil) (defs nil) cv)
    (while todo
      (let ((lc (car todo)))
        (setq todo (cdr todo))
        (unless (member lc done)
          (setq done (cons lc done))
          (setq cv (and (ddGetObj (car lc) (cadr lc) viewName)
                        (dbOpenCellViewByType (car lc) (cadr lc) viewName)))
          (when cv
            (setq defs (cons (list (cadr lc) cv) defs))
            (foreach inst (getq cv instances)
              (let ((mlib (getq inst libName))
                    (mcell (getq inst cellName)))
                ;; only descend into cells that look like subcircuits:
                ;; anything with its own schematic view. Probe with
                ;; ddGetObj first: dbOpenCellViewByType warns in the
                ;; CIW for every PDK primitive that has none
                (when (and mlib mcell
                           (ddGetObj mlib mcell viewName)
                           (dbOpenCellViewByType mlib mcell viewName))
                  (setq todo (cons (list mlib mcell) todo)))))))))
    (reverse defs)))

(procedure (CInHintsExport libName cellName viewName outPath)
  (let ((out (outfile outPath "w")) (defs nil))
    (unless out (error "cin_hints: cannot write %s" outPath))
    (setq defs (CInHCollectDefs libName cellName viewName))
    (fprintf out "{\n  \"hints_version\": \"0.1\",\n")
    (fprintf out "  \"source\": {\"lib\": %s, \"cell\": %s, \"view\": %s},\n"
             (CInJsonStr libName) (CInJsonStr cellName)
             (CInJsonStr viewName))
    (fprintf out "  \"top\": %s,\n" (CInJsonStr cellName))
    (fprintf out "  \"definitions\": {\n")
    (let ((frags nil))
      (foreach def defs
        (let ((sport (outstring)))
          (CInHEmitDef (cadr def) (car def) sport)
          (setq frags (cons (getOutstring sport) frags))))
      (fprintf out "%s\n" (buildString (reverse frags) ",\n")))
    (fprintf out "  }\n}\n")
    (close out)
    (printf "cin_hints: wrote %s (%L definition(s))\n"
            outPath (length defs))
    outPath))
