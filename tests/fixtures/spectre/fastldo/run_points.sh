#!/bin/bash
# Capture the fastldo fixture: one psf dir per (IL, CL) operating point, each
# with dcOp, dcOpInfo and the stb sweep at the main-loop probe IPRB0.
#   module load project/ICDESIGN pdk=SKY130; ./run_points.sh
module load project/ICDESIGN pdk=SKY130 >/dev/null 2>&1
cd ~/CircuitInsight/ldo_sky130
dirs=""
for pt in 20u:1p 20u:10n 20u:1u 1m:1p 1m:100n 1m:1u 100m:1p 100m:1u; do
  il=${pt%%:*}; cl=${pt##*:}
  sed "s/ il=1m / il=$il /; s/ cl=1p / cl=$cl /" tb_fastldo.scs > _tb_${il}_${cl}.scs
  rm -rf psf_${il}_${cl}
  timeout 900 spectre _tb_${il}_${cl}.scs -format psfascii -raw ./psf_${il}_${cl} +log run_${il}_${cl}.log > /dev/null 2>&1
  echo "=== il=$il cl=$cl exit $? ==="
  grep -iE "error" run_${il}_${cl}.log | head -3
  rm -f psf_${il}_${cl}/logFile
  dirs="$dirs psf_${il}_${cl}"
done
tar czf psf8.tgz $dirs
