#!/bin/bash
# Capture the nmc3 behavioral fixture: stb at each of the three nested-loop
# probes (outer IPRB0, Cm2 IPRB2, Cm1 IPRB1) + one dcOpInfo. Run on the
# Spectre host (any pdk works; the deck is PDK-free):
#   module load project/ICDESIGN pdk=SKY130
#   ./run_nmc3.sh
set -u
module load project/ICDESIGN pdk=SKY130
cd ~/CircuitInsight/nmc3

run() {  # <probe> <psfdir>
  sed "s/probe=IPRB0/probe=$1/" tb_nmc3.scs > _tb_$1.scs
  spectre "_tb_$1.scs" -format psfascii -raw "./$2" +log "$1.log" >/dev/null 2>&1
  echo "=== $1 exit: $? ==="
  grep -iE "ERROR" "$1.log" | head -8
  grep -A5 -i "phaseMargin\|gainMargin\|stb_state" "$1.log" | head -12
}

run IPRB0 psf          # outer feedback loop (canonical: carries dcOpInfo)
run IPRB2 psf_cm2      # Cm2 outer-Miller loop
run IPRB1 psf_cm1      # Cm1 inner-Miller loop
echo "=== done ==="
ls -la psf psf_cm2 psf_cm1
