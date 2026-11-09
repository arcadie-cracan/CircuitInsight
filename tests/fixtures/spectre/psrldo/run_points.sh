#!/bin/bash
# Capture the psrldo fixture: one psf dir per (IL, CL) operating point, each
# with dcOp, dcOpInfo, the stb sweep at the main-loop probe IPRB0 and the
# supply-driven ac sweep (VIN mag=1) for the PSR. The two *_noenh points
# have the PSR enhancer's Cin removed (cin=0), the paper's "without PSR
# Enhancer" comparison of its Fig. 14.
#   module load project/ICDESIGN pdk=SKY130; ./run_points.sh
module load project/ICDESIGN pdk=SKY130 >/dev/null 2>&1
cd ~/CircuitInsight/psr_sky130
dirs=""
run() {
  il=$1; cl=$2; tag=$3; extra=$4
  sed "s/ il=1m / il=$il /; s/ cl=100p / cl=$cl /; $extra" tb_psrldo.scs > _tb_${il}_${cl}${tag}.scs
  rm -rf psf_${il}_${cl}${tag}
  timeout 900 spectre _tb_${il}_${cl}${tag}.scs -format psfascii -raw ./psf_${il}_${cl}${tag} +log run_${il}_${cl}${tag}.log > /dev/null 2>&1
  echo "=== il=$il cl=$cl $tag exit $? ==="
  grep -iE "error" run_${il}_${cl}${tag}.log | head -3
  rm -f psf_${il}_${cl}${tag}/logFile
  dirs="$dirs psf_${il}_${cl}${tag}"
}
for il in 1u 10u 100u 1m 10m 50m; do
  for cl in 10p 1n; do run $il $cl "" ""; done
done
run 1m 100p "" ""
run 50m 100p "" ""
run 1m 100p _noenh "s/ cin=1p / cin=0 /"
run 50m 100p _noenh "s/ cin=1p / cin=0 /"
tar czf psf16.tgz $dirs
echo ALLDONE
