# CircuitInsight

[![tests](https://github.com/arcadie-cracan/CircuitInsight/actions/workflows/ci.yml/badge.svg)](https://github.com/arcadie-cracan/CircuitInsight/actions/workflows/ci.yml)

Symbolic small-signal circuit analysis driven by the simulator's own operating
point. This public snapshot ships the tool and its installation instructions;
full documentation will follow.

## Requirements

- Python ≥ 3.11
- Runtime dependencies (installed automatically): NumPy, SymPy, gmpy2.
  On conda/miniforge, `conda install -c conda-forge gmpy2` first is the
  smoothest way to get gmpy2's GMP backend.

## Standalone install

```bash
python -m venv ~/venvs/circuitinsight
source ~/venvs/circuitinsight/bin/activate
pip install "circuitinsight[gui] @ git+https://github.com/arcadie-cracan/CircuitInsight.git"
circuitinsight-gui --help
```

Or from an editable checkout, with the test suite (runs entirely from
checked-in fixtures — no EDA tools or licenses required):

```bash
git clone https://github.com/arcadie-cracan/CircuitInsight.git
cd CircuitInsight
pip install -e .[gui,dev]
pytest
```

> The foundry-PDK-derived validation fixtures are being regenerated on the
> open-source SKY130 PDK and will ship here shortly; until then the public
> suite covers the synthetic golden circuits and the µA741 teaching deck.

The GUI's Expression tab uses Qt WebEngine when the PySide6 install provides it
(pip's PySide6 does) and falls back to a matplotlib rendering otherwise.

## Cadence Virtuoso / ADE integration

### 1. Python environment on the simulation host

ADE writes **binary** PSF results, which are read through Cadence's
`cdspythonsrr` wheel — imported by the very interpreter that runs the GUI. The
wheel is built for one specific Python version, so it dictates the
environment's version. Check it first:

```bash
ls $CDSHOME/tools/python/64bit/virtuoso/     # e.g. cdspythonsrr-…-cp311-…whl
```

If miniforge is a shared/read-only depot install, point conda at your home
first (otherwise `mamba create` fails with `Permission denied`):

```bash
conda config --add pkgs_dirs ~/.conda/pkgs
conda config --add envs_dirs ~/.conda/envs
```

Create the environment (matching the wheel's `cpXY`) and install:

```bash
mamba create -n circuitinsight -c conda-forge \
    python=3.11 pyside6 matplotlib numpy sympy gmpy2
mamba activate circuitinsight
pip install $CDSHOME/tools/python/64bit/virtuoso/cdspythonsrr-*.whl
pip install "git+https://github.com/arcadie-cracan/CircuitInsight.git"

python -c "import cdspythonsrr; print('SRR OK')"
circuitinsight-skill-path       # -> <skill-dir>: the .il files ship in the package
```

`cdspythonsrr` needs the Cadence environment for licensing; launched from the
CIW it inherits Virtuoso's, so it just works.

> No wheel? Rerun spectre with `-format psfascii` and skip `cdspythonsrr`
> entirely — the pure-Python backend reads ascii results.

### 2. The launcher wrapper

Virtuoso's `ipcBeginProcess` starts the GUI in a bare shell — no PATH and no
conda activation. Wrap it, building the wrapper *with the env active* so the
conda base is baked in (do not hardcode `~/miniforge3`; on a shared install
that path is wrong):

```bash
mkdir -p ~/bin
CONDA_BASE="$(conda info --base)"
cat > ~/bin/ci-gui <<EOF
#!/usr/bin/env bash
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate circuitinsight
unset SESSION_MANAGER        # silences a benign XSMP warning over SSH/VNC
exec circuitinsight-gui "\$@"
EOF
chmod +x ~/bin/ci-gui
```

Prove it self-activates from a shell with **no** conda env — exactly the
condition `ipcBeginProcess` creates:

```bash
conda deactivate            # repeat until no env shows in the prompt
~/bin/ci-gui --help         # must still print the usage
```

PySide6 also needs a display (`ssh -X` or VNC).

### 3. Hook into Virtuoso (two lines in `.cdsinit`)

```lisp
CInGuiCmd = "/home/you/bin/ci-gui"          ; ABSOLUTE path — ipc gets no PATH
load("<skill-dir>/cin_init.il")             ; <skill-dir> from circuitinsight-skill-path
```

`cin_init.il` locates its own directory, loads the exporter and launcher from
beside itself, and registers a callback so the **CircuitInsight** menu appears
on ADE windows as they open. Optional overrides: `CInGround` (default
`("0" "gnd!")`), `CInSchematicView` (default `"schematic"`), `CInSimDir`,
`CInMenuDelay`.

### 4. Use

Run **dc + ac** in ADE, then **CircuitInsight → Analyze current run…** — or,
from the CIW:

```lisp
CInAnalyze()                          ; dispatches on the focused window
CInAnalyze(?psf "/path/to/psf")       ; override the detected results dir
```

## License

MIT — see [LICENSE](LICENSE).
