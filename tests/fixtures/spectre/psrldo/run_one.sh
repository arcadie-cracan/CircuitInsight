#!/bin/bash
# one operating point of tb_psrldo.scs: ./run_one.sh <il> <cl> [extra sed]
module load project/ICDESIGN pdk=SKY130 >/dev/null 2>&1
cd ~/CircuitInsight/psr_sky130
il=$1; cl=$2; tag=${3:-}
sed "s/ il=1m / il=$il /; s/ cl=100p / cl=$cl /" tb_psrldo.scs > _tb_${il}_${cl}${tag}.scs
if [ -n "$4" ]; then sed -i "$4" _tb_${il}_${cl}${tag}.scs; fi
rm -rf psf_${il}_${cl}${tag}
timeout 900 spectre _tb_${il}_${cl}${tag}.scs -format psfascii -raw ./psf_${il}_${cl}${tag} +log run_${il}_${cl}${tag}.log > /dev/null 2>&1
echo "=== il=$il cl=$cl $tag exit $? ==="
grep -iE "error|warning" run_${il}_${cl}${tag}.log | head -5
grep -E "^\s*(out|gate|vb|va1|va2|a|b|c|s5|s19|bp|tail|hp|fb)\s" spectre.dc | tr '\n' ' '; echo
rm -f psf_${il}_${cl}${tag}/logFile
