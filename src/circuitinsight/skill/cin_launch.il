;; cin_launch.il — one-click "analyze this schematic in CircuitInsight" (M9)
;;
;; Loose coupling (docs/gui-virtuoso-integration-plan.md, phase 1): export the
;; current schematic to CIN with cin_export.il, then launch the standalone GUI
;; on that CIN + a chosen psf results directory. The GUI is a separate process
;; (its own venv) — nothing here depends on the app, and the app never depends
;; on SKILL/Cadence.
;;
;; Requires: cin_export.il loaded first; the `circuitinsight[gui]` console
;; script `circuitinsight-gui` reachable on PATH (or set CInGuiCmd to its
;; absolute path / "python -m circuitinsight.gui").
;;
;; Install (in the CIW or .cdsinit) — these .il files ship with the package;
;; print their directory in a shell with `circuitinsight-skill-path`, then use
;; that <skill-dir> below:
;;   CInGuiCmd = "/abs/path/to/gui-launcher"   ; ipcBeginProcess gets no PATH
;;   CInSimDir = "/abs/path/to/ADE/simulation" ; for psf auto-detection
;;   load("<skill-dir>/cin_export.il")
;;   load("<skill-dir>/cin_launch.il")
;;   CInInstallMenu()          ; with the SCHEMATIC window focused
;;   CInInstallAdeMenu()       ; with the ADE window focused
;;
;; Or call directly — CInAnalyze() dispatches on the focused window:
;;   CInAnalyze()              ; schematic -> CInLaunchCurrent, ADE -> CInLaunchAde
;;
;; Either path needs an ADE run to exist: the schematic gives the topology, the
;; ADE session gives the operating point. Both are found automatically; pass
;; ?psf "…" to override the detected results directory.
;;
;; ------------------------------------------------------------------ config
(unless (boundp 'CInGuiCmd)
  ;; the [gui] entry point; override with an absolute venv path if not on PATH
  (setq CInGuiCmd "circuitinsight-gui"))
(unless (boundp 'CInTmpDir)
  (setq CInTmpDir (or (getShellEnvVar "TMPDIR") "/tmp")))
(unless (boundp 'CInGround)
  (setq CInGround (list "0" "gnd!")))      ; same convention as CInExport
(unless (boundp 'CInSimDir)
  ;; nil => query ADE for it (see CInAdeSimDir). Set explicitly only to override.
  (setq CInSimDir nil))
(unless (boundp 'CInMenuDelay)
  ;; Delay before installing the ADE menu, in TENTHS OF A SECOND -- that is
  ;; hiRegTimer's unit (x_tenthsofSeconds), not ms and not seconds. 20 = 2 s.
  ;; The session-creation callback fires before the session is active and before
  ;; its window exists, so we let it settle first.
  ;; Raise this if the menu reports "no window for ADE session".
  (setq CInMenuDelay 20))
(unless (boundp 'CInSchematicView)
  ;; the view CInExport walks when launching from ADE: the ADE session's own view
  ;; (maestro) is the simulation setup, not the topology
  (setq CInSchematicView "schematic"))

;; ------------------------------------------------- ADE psf auto-detection
;; ADE Explorer/Assembler writes results to
;;   <CInSimDir>/<lib>/<cell>/<view>/results/<view>/<history>/<point>/<test>/psf
;; Rather than trust that shape, walk the run's tree and keep only psf dirs that
;; actually carry OP data (dcOpInfo) -- that rejects ADE's .tmpADEDir_* scratch
;; dirs and empty psf shells, and is exactly the data CircuitInsight needs.

(procedure (CInAdeSimDir)
  "ADE's simulation directory, absolute; nil if it cannot be determined.

It is not session state -- it is the .cdsenv variable asimenv.startup/projectDir,
and it is typically RELATIVE (e.g. \"./simulation\"), resolved against the
directory Virtuoso was started in. Hence envGetVal + getWorkingDir, not any axl*
accessor -- which is why no session query ever found it."
  (let ((d (car (errset (envGetVal "asimenv.startup" "projectDir")))))
    (when (and d (stringp d) (nequal d ""))
      (unless (rexMatchp "^[/~]" d)
        (setq d (strcat (getWorkingDir) "/" d)))
      (or (car (errset (simplifyFilename d))) d))))

(procedure (CInSimDirEff)
  "The simulation directory actually used: explicit override, else ADE's."
  (or CInSimDir (CInAdeSimDir)))

(procedure (CInPsfHasOp d)
  (exists f (getDirFiles d) (rexMatchp "dcOpInfo" f)))

(procedure (CInFindPsf root)
  "Recursively collect psf directories under root that contain OP data."
  (let ((out nil))
    (when (and root (isDir root))
      (foreach f (getDirFiles root)
        (unless (member f (list "." ".."))
          (let ((p (strcat root "/" f)))
            (when (and (isDir p) (null (rexMatchp "tmpADEDir" p)))
              (if (and (equal f "psf") (CInPsfHasOp p))
                  (setq out (cons p out))
                  (setq out (append out (CInFindPsf p)))))))))
    out))

(procedure (CInFindAdeSession @optional (cell nil))
  "An open ADE session (optionally one whose design cell matches `cell`)."
  (let ((found nil))
    (foreach w (car (errset (hiGetWindowList)))
      (unless found
        (let ((s (car (errset (axlGetWindowSession w)))))
          (when (and s (or (null cell)
                           (equal (car (errset (axlGetSessionCellName s))) cell)))
            (setq found s)))))
    found))

(procedure (CInAdeCalcPsf @optional (cell nil))
  "psf dir straight from ADE: asiGetCalcResultsDir() names the selected run's
results directory, and the psf is right underneath it. One call -- no simulation
directory, no lib/cell/view/history path assembly, no tree walk.

It takes no session argument, though: it is the *calculator's* current results
dir, global to the CIW. With several ADE windows open it can belong to another
design, so when the caller knows the cell we require the path to mention it
rather than trust a global."
  (let (d psf)
    (setq d (car (errset (asiGetCalcResultsDir))))
    (when (and d (stringp d))
      (setq psf (strcat d "/psf"))
      (cond
        ((and cell (null (rexMatchp cell d)))
         (printf "CInLaunch: ADE results dir is %s, which is not cell %s -- ignoring it\n"
                 d cell)
         nil)
        ((null (and (isDir psf) (CInPsfHasOp psf)))
         (printf "CInLaunch: %s carries no OP data (dcOpInfo)\n" psf)
         nil)
        (t
         (printf "CInLaunch: psf from ADE -> %s\n" psf)
         psf)))))

(procedure (CInAdePsfDirByPath @optional (sess nil))
  "Fallback: rebuild the psf path from the session + simulation directory, and
walk it for a psf dir carrying OP data. Used when asiGetCalcResultsDir gives us
nothing usable (no run selected in the calculator, or a foreign design).

Returns nil (-> caller errors) when the run has several result sets, e.g.
multiple corners: silently joining OP data from one corner into another run's
topology would be a correctness bug, not a convenience."
  (let (lib cell view hist sim base hits)
    (setq sess (or sess (car (errset (axlGetWindowSession (hiGetCurrentWindow))))))
    (when sess
      (setq lib  (car (errset (axlGetSessionLibName sess))))
      (setq cell (car (errset (axlGetSessionCellName sess))))
      (setq view (car (errset (axlGetSessionViewName sess))))
      (setq hist (car (errset (axlGetHistoryName (axlGetCurrentHistory sess)))))
      (setq sim  (CInSimDirEff))
      (unless sim
        (printf "CInLaunch: no ADE simulation directory; set CInSimDir\n"))
      (when (and lib cell view hist sim)
        (setq base (sprintf nil "%s/%s/%s/%s/results/%s/%s"
                            sim lib cell view view hist))
        (setq hits (CInFindPsf base))
        (cond
          ((null hits)
           (printf "CInLaunch: no psf with OP data under %s\n" base)
           nil)
          ((equal (length hits) 1)
           (printf "CInLaunch: psf from ADE run %s -> %s\n" hist (car hits))
           (car hits))
          (t
           (printf "CInLaunch: %d result sets under %s (corners?) -- please pick one\n"
                   (length hits) base)
           (foreach h hits (printf "    %s\n" h))
           nil))))))

(procedure (CInAdePsfDir @optional (sess nil))
  "psf dir of the current ADE run, or nil. Asks ADE first (asiGetCalcResultsDir),
then falls back to rebuilding the path from the session + simulation directory."
  (let (cell)
    (setq sess (or sess (car (errset (axlGetWindowSession (hiGetCurrentWindow))))))
    (setq cell (car (errset (axlGetSessionCellName sess))))
    (or (CInAdeCalcPsf cell)
        (CInAdePsfDirByPath sess))))

;; --------------------------------------------------------------- launch
(defun CInLaunchCurrent (@key (psf nil))
  "Export the current schematic cellview to CIN and launch the CircuitInsight
GUI on it. If psf is nil, prompt for the ADE/Maestro results directory."
  (let (win cv lib cell view cin cmd)
    (setq win (hiGetCurrentWindow))
    (unless win (error "CInLaunch: no current window"))
    (setq cv (geGetEditCellView win))
    (unless cv (error "CInLaunch: the current window has no cellview"))
    ;; SKILL's setq takes exactly one pair -- no Common-Lisp multi-assignment
    (setq lib  cv~>libName)
    (setq cell cv~>cellName)
    (setq view cv~>viewName)
    (setq cin (sprintf nil "%s/%s__%s.cin.json" CInTmpDir cell view))

    ;; 1. capture topology (hard error inside CInExport on any unmapped device)
    (printf "CInLaunch: exporting %s/%s/%s -> %s\n" lib cell view cin)
    (CInExport lib cell view cin ?ground CInGround)

    ;; 2. resolve the psf results directory from the live ADE run for this cell.
    ;;    Pulling it from the session that just ran is what keeps the CIN and the
    ;;    OP data from coming out of different simulations.
    (unless psf
      (setq psf (CInAdePsfDir (CInFindAdeSession cell))))
    (unless psf
      (error (strcat "CInLaunch: could not determine the psf directory from an "
                     "open ADE session -- pass it explicitly:\n"
                     "  CInLaunchCurrent(?psf \"/path/to/run/psf\")")))

    ;; 3. launch the standalone GUI as a detached process
    (setq cmd (sprintf nil "%s --cin '%s' --psf '%s'" CInGuiCmd cin psf))
    (printf "CInLaunch: %s\n" cmd)
    (ipcBeginProcess cmd)
    t))

;; ---------------------------------------------------- launch from ADE
(defun CInLaunchAde (@key (psf nil))
  "From an ADE Explorer/Assembler window: export the design's *schematic* to CIN
and launch the GUI on the current run's results. The ADE view (maestro) is the
simulation setup, not the topology, so CInExport walks CInSchematicView instead.

The schematic does NOT need to be open: CInExport reads the cellview from the
design database (dbOpenCellViewByType), not from a window. This is why the ADE
menu is the better entry point -- both the topology and the operating point are
resolved from the one session, so they cannot come from different designs."
  (let (sess lib cell cin cmd)
    (setq sess (or (car (errset (axlGetWindowSession (hiGetCurrentWindow))))
                   (CInFindAdeSession)))
    (unless sess (error "CInLaunchAde: no ADE session (is an ADE window open?)"))
    (setq lib  (axlGetSessionLibName sess))
    (setq cell (axlGetSessionCellName sess))
    (setq cin (sprintf nil "%s/%s__%s.cin.json" CInTmpDir cell CInSchematicView))

    (printf "CInLaunchAde: exporting %s/%s/%s -> %s\n"
            lib cell CInSchematicView cin)
    (CInExport lib cell CInSchematicView cin ?ground CInGround)

    (unless psf (setq psf (CInAdePsfDir sess)))
    (unless psf
      (error (strcat "CInLaunchAde: could not determine the psf directory for "
                     "this run -- pass it explicitly:\n"
                     "  CInLaunchAde(?psf \"/path/to/run/psf\")")))

    (setq cmd (sprintf nil "%s --cin '%s' --psf '%s'" CInGuiCmd cin psf))
    (printf "CInLaunchAde: %s\n" cmd)
    (ipcBeginProcess cmd)
    t))

;; ------------------------------------------------------- single entry point
(defun CInAnalyze (@key (psf nil))
  "Analyze from whichever window is focused: an ADE Explorer/Assembler window
(-> CInLaunchAde) or a schematic (-> CInLaunchCurrent). Both menus call this, so
'Analyze' does the right thing without the user tracking which window they are
in -- the schematic path needs an open ADE session for the psf anyway, and the
ADE path finds the schematic from the session."
  (let ((w (hiGetCurrentWindow)))
    (unless w (error "CInAnalyze: no current window"))
    (if (car (errset (axlGetWindowSession w)))
        (CInLaunchAde ?psf psf)
        (CInLaunchCurrent ?psf psf))))

;; ----------------------------------------------------------------- menus
;; Both menus attach to the CURRENT window. Virtuoso's design-editor trigger API
;; (deRegUserTriggerFunctions) does not exist in IC23.1, and ADE windows have no
;; equivalent, so installation is explicit per window rather than automatic --
;; call these with the window you want the menu on focused. hiCreateMenu binds
;; the menu to its symbol, so create once and reuse on later windows.

(defun CInInstallMenu (@rest _)
  "Add a 'CircuitInsight' menu to the CURRENT schematic window."
  (let ((w (hiGetCurrentWindow)))
    (unless w (error "CInInstallMenu: no current window"))
    (unless (and (boundp 'CInMenu) CInMenu)
      (hiCreateMenu
        'CInMenu "CircuitInsight"
        (list (hiCreateMenuItem
                ?name 'CInAnalyze
                ?itemText "Analyze current schematic…"
                ?callback "CInAnalyze()"))))
    (hiInsertBannerMenu w CInMenu 99)
    (printf "CInLaunch: menu added to %L\n" w)
    t))

(procedure (CInAdeMenuObj)
  "The CircuitInsight ADE menu -- created once, reused on every window."
  (unless (and (boundp 'CInAdeMenu) CInAdeMenu)
    (hiCreateMenu
      'CInAdeMenu "CircuitInsight"
      (list (hiCreateMenuItem
              ?name 'CInAdeAnalyze
              ?itemText "Analyze current run…"
              ?callback "CInAnalyze()"))))
  CInAdeMenu)

(procedure (CInAdeWindow sess)
  "The window whose ADE session is `sess`, or nil.

NOT via axlGetSessionWindowNumber: that returns a number in ADE's own numbering,
not a Virtuoso window id (it reports 3 for a session living in window:5), so
window() on it finds the wrong window or none. Invert the mapping we have
verified instead -- axlGetWindowSession over the open windows."
  (let ((found nil))
    (foreach w (car (errset (hiGetWindowList)))
      (unless found
        (when (equal (car (errset (axlGetWindowSession w))) sess)
          (setq found w))))
    found))

(defun CInInstallAdeMenu (@rest _)
  "Add a 'CircuitInsight' menu to the CURRENT ADE Explorer/Assembler window."
  (let ((w (hiGetCurrentWindow)))
    (unless w (error "CInInstallAdeMenu: no current window"))
    (hiInsertBannerMenu w (CInAdeMenuObj) 99)
    (printf "CInLaunchAde: menu added to %L\n" w)
    t))

(defun CInInstallAdeMenuFor (sess @optional (retries 15))
  "Deferred menu install for ADE session `sess` (see CInAdeSessionCb).

RETRIES, not one-shot: a FRESHLY CREATED maestro view spends longer than
one CInMenuDelay inside the creation wizard, so the session exists but
its window does not yet -- a single-shot install fired into the void and
the menu only ever appeared after the view was closed and reopened. Keep
re-arming the timer (retries x CInMenuDelay, default 15 x 2 s = 30 s)
until the window shows up; the double-insert guard makes late retries
harmless when the user opened the menu path some other way."
  (let ((w (CInAdeWindow sess)))
    (cond
      ((null w)
       (if (greaterp retries 0)
           (errset (hiRegTimer
                     (sprintf nil "CInInstallAdeMenuFor(\"%s\" %d)"
                              sess (difference retries 1))
                     CInMenuDelay))
           (printf (strcat "CircuitInsight: no window for ADE session %s "
                           "after retries -- call CInInstallAdeMenu() "
                           "there, or raise CInMenuDelay\n")
                   sess))
       nil)
      ((member 'CInAdeMenu (car (errset (hiGetBannerMenus w))))
       t)                                   ; already installed; stay quiet
      (t
       (hiInsertBannerMenu w (CInAdeMenuObj) 99)
       (printf "CircuitInsight: menu added to ADE session %s (%L)\n" sess w)
       t))))

(defun CInAdeSessionCb (@rest args)
  "Callback for axlSessionRegisterCreationCallback (registered by cin_init.il).

It fires at the START of session creation: the session is not active yet and has
no window, so touching it here yields ADE-1600 and hiGetCurrentWindow is nil.
The menu therefore cannot be installed now -- defer it with hiRegTimer and let
CInInstallAdeMenuFor run once the session is live.

hiRegTimer's delay is in TENTHS OF A SECOND (x_tenthsofSeconds). Getting that
unit wrong fails silently -- the install simply sits in the future -- so it is
worth stating plainly here.

Never allowed to break session creation, so it is fully guarded."
  (let ((sess (car args)))
    (when (and sess (stringp sess))
      (errset (hiRegTimer (sprintf nil "CInInstallAdeMenuFor(\"%s\")" sess)
                          CInMenuDelay))))
  t)

;; -------------------------------------------------- stb probe discovery
;; The GUI discovers the stb analysis's designated probe BY ITSELF, from the
;; run's psfascii header or the run netlist beside the psf directory (the
;; record of the analyses as defined) -- see SpectreRun.stb_probe(). The
;; launcher therefore passes nothing extra; when a run carries an stb
;; analysis the GUI opens in the Loop-gain bench with that probe selected.
;; To force a different vsource: append --probe <inst> to CInGuiCmd, or use
;; the GUI's probe combo (ANY vsource qualifies as a Tian probe).

;; ----------------------------------------------------------- smoke test
;; Manual, on <sim-host> (needs the ICDESIGN/SKY130 env + a display):
;;   1. Open a testbench schematic (e.g. worklib/tb_ota5t/schematic) and
;;      run its ADE/Maestro dc + ac so a psf results dir exists.
;;   2. In the CIW:  load both .il files, then CInInstallMenu().
;;   3. From the schematic's CircuitInsight menu, pick "Analyze current
;;      schematic…", choose the run's psf directory when prompted.
;;   4. The GUI opens preloaded; Suggest matches, Rank, Solve — the Bode should
;;      overlay the ADE ac. Any device without OP data is a hard error (name
;;      correlation) reported by CInExport / the join, per plan section 3.2.
;;
;; Direct (no menu):  CInLaunchCurrent(?psf "/path/to/run/psf")
