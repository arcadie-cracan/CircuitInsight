#!/bin/bash
# Capture the nmc3d fixture: stb at the DM probe (FDPRB.IPRB_DM, flattened to
# IPRB_DM) + dcOpInfo. Run on the Spectre host (any pdk; the deck is PDK-free):
#   module load project/ICDESIGN pdk=SKY130
#   ./run_nmc3d.sh
set -u
module load project/ICDESIGN pdk=SKY130
cd ~/CircuitInsight/nmc3d
spectre tb_nmc3d.scs -format psfascii -raw ./psf +log nmc3d.log >/dev/null 2>&1
echo "=== exit: $? ==="
grep -iE "ERROR" nmc3d.log | head -8
grep -A5 -i "phaseMargin|gainMargin|stb_state" nmc3d.log | head -12
ls -la psf
