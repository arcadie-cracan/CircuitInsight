;; cin_xprobe.il — schematic cross-probe helpers (M10)
;;
;; Called from the CircuitInsight GUI over skillbridge (see
;; circuitinsight/virtuoso/xprobe.py). The Python side decides WHICH device a
;; clicked symbol names; everything here is the Cadence-side database work:
;; descend a hierarchical instance path, select it, and make it visible.
;;
;; cin_init.il loads this file, so the one .cdsinit line covers it. To load it
;; by hand into a running session (CIW):
;;   load("<skill-dir>/cin_xprobe.il")
;; Print <skill-dir> with `circuitinsight-skill-path`.
;;
;; The GUI checks CInHighlight is defined before offering the feature, so
;; loading this file is what turns cross-probing on.
;;
;; Paths use the same spelling as the CIN/OP join keys: dot-separated,
;; top-down, e.g. "I0.MN0" is instance MN0 inside instance I0 of the window's
;; cellview. A single segment ("MN0") is a device in the top cellview.
;;
;; ---------------------------------------------------------------------------
;; API facts CONFIRMED on Virtuoso Studio IC23.1 (2026-07-25 live session).
;; Each of these cost a failed cross-probe that still reported success, so they
;; are recorded rather than rediscovered:
;;
;;   geSelectObject(obj)      ONE argument. Passing a window is an arity error
;;                            ("argument #1 should be a database object").
;;                            Returns nil -- not an error -- when the object is
;;                            already selected.
;;   geGetSelectedSet()       NO argument, and reports the CURRENT window's
;;   geDeselectAll()          selection. To act on another window, make it
;;                            current first and restore afterwards.
;;   hiHighlight              DOES NOT EXIST.
;;   hiZoomToSelectionSet     DOES NOT EXIST. There is no auto-zoom; the device
;;                            is selected where it sits and hiRedraw repaints.
;;   inst~>master             is the SYMBOL view (no child instances). Descend
;;                            with dbOpenCellViewByType(lib cell "schematic").
;;   window~>cellView         nil on the CIW -- the cheap test for "is this a
;;                            graphics window", avoiding GE-2067, which is a
;;                            WARNING and so is NOT caught by errset.
;;
;; And the structural one: a sub-cellview's device can only be *seen* selected
;; in a window that displays that cellview, so the highlight appears in the
;; sub-schematic's own tab, not the testbench's.
;; ---------------------------------------------------------------------------

(unless (boundp 'CInXprobeWindow)
  ;; nil => use the current window. Set to a window id to pin cross-probing to
  ;; one schematic while you work in other windows.
  (setq CInXprobeWindow nil))

(procedure (CInXprobeWin)
  "The window paths are resolved FROM (the top of the exported hierarchy)."
  (or CInXprobeWindow (hiGetCurrentWindow)))

(defun CInXprobeHere (@rest _)
  "Pin the current window as the cross-probe ROOT and report it.

Without pinning, paths resolve against whatever window is current -- so
switching to a sub-schematic to look at a highlight makes the next probe fail
against that sub-cellview (I0.MN0 has no meaning inside ota5t). The root must
be the cellview the CIN was exported from, so pin it once and then browse
freely. The GUI calls this when cross-probing is switched on."
  (let ((w (hiGetCurrentWindow)) cv)
    (unless w (error "CInXprobeHere: no current window"))
    (setq cv (CInWindowCv w))
    (unless cv (error "CInXprobeHere: %L is not a schematic window" w))
    (setq CInXprobeWindow w)
    (printf "CInXprobe: root pinned to %L (%s/%s/%s)\n" w
            (getq cv libName) (getq cv cellName) (getq cv viewName))
    t))

(defun CInXprobeUnpin (@rest _)
  "Follow the current window again."
  (setq CInXprobeWindow nil)
  (printf "CInXprobe: root follows the current window\n")
  t)

(procedure (CInFindInst cv name)
  "The instance called `name` in cellview `cv`, or nil.

dbFindAnyInstByName is not available on every release/exhibit, so match on the
instance list -- small cellviews, and it keeps the traversal release-agnostic."
  (let ((hit nil))
    (when cv
      (foreach i cv~>instances
        (unless hit
          (when (equal i~>name name) (setq hit i)))))
    hit))

(procedure (CInSchematicOf inst)
  "The SCHEMATIC cellview of `inst`'s master, or nil.

Not inst~>master: a schematic instance's master is its SYMBOL view, which
holds only graphics and pins, so descending through it dead-ends immediately
(this was the first live-test failure -- I0's master reported 0 instances).
Re-open the same lib/cell as its schematic, which is exactly how cin_export.il
walks the hierarchy when it builds the CIN, so the instance paths this
resolves are the paths the join keys were spelled from."
  (let (m)
    (setq m (car (errset inst~>master)))
    (when m
      (car (errset (dbOpenCellViewByType
                     (getq m libName) (getq m cellName) "schematic" "" "r"))))))

(procedure (CInResolve path @optional (win nil))
  "Resolve a dot-separated instance path to list(containingCellView leafName).
nil when any segment is missing -- a cross-probe that silently highlights the
wrong device would be worse than one that highlights nothing.

Returns the CONTAINING cellview and the leaf's NAME, not a db object: the
object must be looked up again in whichever window actually displays that
cellview (see CInHighlight). A db object from this read-only handle is not the
same object the user's open tab is editing, and selecting it in the wrong
window does nothing visible -- which is exactly what the first live attempt
did, returning t while nothing lit up."
  (let (cv inst segs last_seg)
    (setq win (or win (CInXprobeWin)))
    (unless win (return nil))
    (setq cv (car (errset (geGetEditCellView win))))
    (unless cv (return nil))
    (setq segs (parseString path "."))
    (setq last_seg (car (last segs)))
    (foreach seg segs
      (when cv
        (cond
          ((equal seg last_seg)
           ;; the leaf must exist here, but we hand back the CELLVIEW
           (unless (CInFindInst cv seg) (setq cv nil)))
          (t
           (setq inst (CInFindInst cv seg))
           (setq cv (and inst (CInSchematicOf inst)))))))
    (and cv (list cv last_seg))))

(procedure (CInWindowCv w)
  "The cellview a window edits, or nil for non-graphics windows.

Guarded by the window's own cellView attribute first: geGetEditCellView on the
CIW emits GE-2067 (\"not a graphic editor window\"), and that is a WARNING, so
errset does not swallow it -- it would print on every cross-probe."
  (let ((cv (car (errset (getq w cellView)))))
    (when cv (car (errset (geGetEditCellView w))))))

(procedure (CInWindowFor cv)
  "An open window editing the same lib/cell/view as `cv`, or nil."
  (let ((found nil) wcv)
    (foreach w (car (errset (hiGetWindowList)))
      (unless found
        (setq wcv (CInWindowCv w))
        (when (and wcv
                   (equal (getq wcv libName)  (getq cv libName))
                   (equal (getq wcv cellName) (getq cv cellName))
                   (equal (getq wcv viewName) (getq cv viewName)))
          (setq found w))))
    found))

(procedure (CInOk res)
  "errset returns (value) on success and nil on error -- so a LIST means the
call went through, even when the value itself is nil."
  (and res (listp res)))

(procedure (CInDeselectIn w)
  "Clear window `w`'s selection. Like geGetSelectedSet, geDeselectAll acts on
the CURRENT window, so w is made current for the call and the previous window
restored."
  (let ((prev (hiGetCurrentWindow)) r)
    (when (and w (nequal w prev)) (errset (hiSetCurrentWindow w)))
    (setq r (or (CInOk (errset (geDeselectAll)))
                (CInOk (errset (geDeselectAll w)))))
    (when (and prev (nequal w prev)) (errset (hiSetCurrentWindow prev)))
    r))

(procedure (CInSelectedIn w)
  "The selected set of window `w`.

geGetSelectedSet takes NO window argument (confirmed on IC23.1: passing one is
an arity error) -- it reports the CURRENT window's selection. So `w` is made
current for the read and the previous window restored, rather than reading
whatever the user happens to be looking at and calling it w's."
  (let ((prev (hiGetCurrentWindow)) r)
    (when (and w (nequal w prev)) (errset (hiSetCurrentWindow w)))
    (setq r (errset (geGetSelectedSet)))
    (when (and prev (nequal w prev)) (errset (hiSetCurrentWindow prev)))
    (when (CInOk r) (car r))))

(procedure (CInHighlight paths)
  "Select the given instance paths and bring them into view. `paths` is a list
of dot-separated paths (a single string is also accepted). Returns t when at
least one was selected.

A device below the top level lives in a SUB-cellview, and selecting it only
shows up in a window displaying that cellview -- so each path is selected in
the window that actually edits its containing cell, which for a hierarchical
path is the sub-schematic's own tab, not the testbench. Open that schematic to
see the highlight; we say so rather than opening windows behind the user's
back on every click.

Selection is the highlight: it is what a designer already recognises, it
survives redraws, and geSelectObject needs no graphics-context bookkeeping. We
deselect first, per touched window, so each cross-probe shows exactly the
current set."
  (let ((n 0) (wins nil) res cv leaf w obj)
    (when (stringp paths) (setq paths (list paths)))
    ;; clear the top window even if nothing resolves, so a failed probe does
    ;; not leave the previous selection looking like the current answer
    (when (CInXprobeWin) (CInDeselectIn (CInXprobeWin)))
    (foreach p paths
      (setq res (CInResolve p))
      (cond
        ((null res)
         (printf "CInXprobe: no instance %s in this cellview\n" p))
        (t
         (setq cv (car res))
         (setq leaf (cadr res))
         (setq w (CInWindowFor cv))
         (cond
           ((null w)
            (printf (strcat "CInXprobe: %s is in %s/%s -- open that schematic "
                            "to see it highlighted\n")
                    p (getq cv libName) (getq cv cellName)))
           (t
            ;; look the instance up in the WINDOW's own cellview: a db object
            ;; from another handle on the same cell is not the same object
            (setq obj (CInFindInst (geGetEditCellView w) leaf))
            (when obj
              (unless (member w wins)
                (CInDeselectIn w)
                (setq wins (cons w wins)))
              ;; ONE argument: geSelectObject takes the db object, and infers
              ;; the window from it. Passing the window is an arity error
              ;; ("argument #1 should be a database object") -- confirmed on
              ;; IC23.1, where it silently cost us every highlight.
              (errset (geSelectObject obj))
              (setq n (plus n 1))))))))
    ;; hiZoomToSelectionSet and hiHighlight do NOT exist on IC23.1, so the
    ;; selection is left where it is and only repainted. Opening the
    ;; sub-schematic shows the device selected; we do not yank the view.
    (foreach w wins (errset (hiRedraw w)))
    (greaterp n 0)))

(procedure (CInXprobeDebug path)
  "Print every stage of resolving and selecting `path`. For diagnosing a
cross-probe that reports success but shows nothing."
  (let (res cv leaf w wcv obj sel)
    (printf "--- CInXprobeDebug %s ---\n" path)
    (printf "current window      : %L\n" (CInXprobeWin))
    (printf "  its cellView attr : %L\n"
            (car (errset (getq (CInXprobeWin) cellView))))
    (setq res (CInResolve path))
    (cond
      ((null res) (printf "resolve             : FAILED\n"))
      (t
       (setq cv (car res)) (setq leaf (cadr res))
       (printf "resolved to         : %s/%s/%s  leaf=%s\n"
               (getq cv libName) (getq cv cellName) (getq cv viewName) leaf)
       (printf "open windows        : %L\n" (car (errset (hiGetWindowList))))
       (foreach w2 (car (errset (hiGetWindowList)))
         (setq wcv (CInWindowCv w2))
         (printf "   %L -> %s\n" w2
                 (if wcv (sprintf nil "%s/%s/%s" (getq wcv libName)
                                  (getq wcv cellName) (getq wcv viewName))
                     "(no cellview)")))
       (setq w (CInWindowFor cv))
       (printf "window for that cv  : %L\n" w)
       (when w
         (setq obj (CInFindInst (geGetEditCellView w) leaf))
         (printf "instance in it      : %L\n" obj)
         (when obj
           ;; errset returns (value) on success, nil on error -- so (nil) here
           ;; means the call worked and simply had nothing to do, i.e. the
           ;; object was already selected
           (printf "geSelectObject(obj) : %L\n" (errset (geSelectObject obj)))
           (setq sel (CInSelectedIn w))
           (printf "selected set now    : %L\n" (and sel (getq sel name)))
           (printf "hiRedraw            : %L\n" (errset (hiRedraw w)))))))
    t))

(procedure (CInHighlightClear)
  "Drop the cross-probe selection."
  (let ((win (CInXprobeWin)))
    (when win (errset (geDeselectAll win)) t)))

(procedure (CInSelection)
  "The instance names currently selected in the cross-probe window, as a list
of strings -- lets the GUI adopt the designer's schematic selection.

Names are returned as the top cellview sees them (one segment). Descending
selections made inside a hierarchy edit are reported by their local name; the
GUI matches against its own instance list, so an unmatched name is ignored
rather than guessed at."
  (let ((win (CInXprobeWin)) (out nil))
    (when win
      (foreach obj (CInSelectedIn win)
        (let ((nm (car (errset obj~>name))))
          (when (and nm (stringp nm)) (setq out (cons nm out))))))
    (reverse out)))

;; ----------------------------------------------------------- smoke test
;; Manual, on <sim-host> (needs a display + the skillbridge server):
;;   1. Start Virtuoso with the skillbridge server (see its docs:
;;      `python -m skillbridge` prints the .il to load).
;;   2. In the CIW: load("<skill-dir>/cin_xprobe.il")
;;   3. Open the testbench schematic; keep that window current.
;;   4. In the CIW, check the traversal directly before involving Python:
;;        CInHighlight("I0.MN0")          ; => t, MN0 selected inside I0
;;        CInHighlight(list("I0.MN0" "I0.MN1"))   ; several at once
;;        CInSelection()                  ; => ("MN0") with it selected
;;        CInHighlightClear()
;;      (Do not write `CInHighlight(list "I0.MN0")`: that mixes the C-style
;;      call with a Lisp-style argument, and SKILL then reads `list` as an
;;      unbound VARIABLE. Either quote the whole form -- (CInHighlight (list
;;      "I0.MN0")) -- or call list() C-style as above.)
;;   5. Launch the GUI (CircuitInsight menu), then click a g_m symbol in the
;;      Expression tab or a row in the Devices table: the device should select
;;      in the schematic.
