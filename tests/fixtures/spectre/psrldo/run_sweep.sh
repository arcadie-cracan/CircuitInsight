#!/bin/bash
# parametric sweep of tb_psrldo.scs at one load point:
#   ./run_sweep.sh <il> <cl> <param> "<v1> <v2> ..." [extra sed on the deck]
# writes psw_<il>_<cl>_<param>/swp-NNN_{ac.ac,stb.margin.stb,dcOp.dc}
module load project/ICDESIGN pdk=SKY130 >/dev/null 2>&1
cd ~/CircuitInsight/psr_sky130
il=$1; cl=$2; par=$3; vals=$(echo $4 | tr ' ' ' ')
out=psw_${il}_${cl}_${par}
sed "s/ il=1m / il=$il /; s/ cl=100p / cl=$cl /" tb_psrldo.scs | grep -vE "^(dcOp|dcOpInfo|stb|ac) " > _sw.scs
if [ -n "$5" ]; then sed -i "$5" _sw.scs; fi
cat >> _sw.scs <<EOT
swp sweep param=$par values=[$vals] {
    dcOp dc write="spectre.dc" maxiters=150
    stb stb start=1 stop=10G dec=20 probe=IPRB0
    ac ac start=1 stop=1G dec=20
}
EOT
rm -rf $out
timeout 1800 spectre _sw.scs -format psfascii -raw ./$out +log sw_${il}_${cl}_${par}.log > /dev/null 2>&1
echo "=== sweep $par at il=$il cl=$cl exit $? ==="
grep -iE "error" sw_${il}_${cl}_${par}.log | head -3
rm -f $out/logFile $out/*.stb.stb 2>/dev/null
ls $out | head -4
tar czf $out.tgz $out
