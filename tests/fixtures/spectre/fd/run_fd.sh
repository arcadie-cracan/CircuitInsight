#!/bin/bash
# run one fdota bench: ./run_fd.sh dm|cm|cmfb|dmref|cmref
set -u
module load project/ICDESIGN pdk=SKY130
cd ~/CircuitInsight/fdota
v="$1"
spectre "tb_fdota_stb_${v}.scs" -format psfascii -raw "./psf_${v}" +log "${v}.log" >/dev/null 2>&1
echo "=== exit: $? ==="
grep -E "ERROR|WARNING" "${v}.log" | head -20
echo "=== margins ==="
grep -A6 -i "stability\|loopGain\|margin" "${v}.log" | head -30
