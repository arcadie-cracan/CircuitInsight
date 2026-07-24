;; cin_init.il — one-line bootstrap for .cdsinit
;;
;;   load("<skill-dir>/cin_init.il")        ; <skill-dir> = `circuitinsight-skill-path`
;;
;; Loads the exporter + launcher from this file's OWN directory (so the path in
;; .cdsinit is the only thing you ever have to write down), then arranges for the
;; CircuitInsight menu to appear on ADE windows as they open.
;;
;; Configure BEFORE the load line if the defaults don't fit:
;;   CInGuiCmd   = "/abs/path/to/gui-launcher"  ; ipcBeginProcess inherits no PATH,
;;                                              ; and no conda activation -- use a
;;                                              ; wrapper, see docs/m9-smoke-test.md
;;   CInSkillDir = "/abs/path/to/skill"         ; only if self-location fails
;;   CInGround   = list("0" "gnd!")             ; ground nets
;;
;; Everything else (the schematic view, the psf results directory) is discovered
;; from the ADE session at analyze time.

(let (self dir)
  ;; ---------------------------------------------------------------- locate self
  ;; get_filename(piport) names the file currently being loaded.
  (setq self (car (errset (get_filename piport))))
  (setq dir
    (or (and (boundp 'CInSkillDir) CInSkillDir)
        (and self (stringp self)
             (car (errset (pcreReplace (pcreCompile "/[^/]*$") self "" 1))))))
  (unless (and dir (isDir dir))
    (error (strcat "cin_init: cannot locate the CircuitInsight SKILL directory. "
                   "Set CInSkillDir (run `circuitinsight-skill-path` in a shell) "
                   "before loading this file.")))
  (setq CInSkillDir dir)

  ;; ------------------------------------------------------------------ the code
  (load (strcat dir "/cin_export.il"))
  (load (strcat dir "/cin_launch.il"))

  ;; ----------------------------------------------------------------- the menu
  ;; .cdsinit runs before any window exists, so we cannot install a menu now.
  ;; Hook ADE session creation instead. (Virtuoso's design-editor trigger API,
  ;; deRegUserTriggerFunctions, does not exist in IC23.1 -- so schematic windows
  ;; stay manual, via CInInstallMenu(). The ADE menu is the better entry point
  ;; anyway: it finds both the schematic and the results from the one session.)
  (if (and (isCallable 'axlSessionRegisterCreationCallback)
           (car (errset (axlSessionRegisterCreationCallback 'CInAdeSessionCb))))
      (printf "CircuitInsight: loaded from %s -- the menu appears on ADE windows as they open\n" dir)
      (printf "CircuitInsight: loaded from %s -- call CInInstallAdeMenu() with an ADE window focused\n" dir))
  t)
