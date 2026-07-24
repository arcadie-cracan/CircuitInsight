#!/bin/bash
set -u
cd ~/CircuitInsight/fdota
echo "=== psf files ==="
ls psf_dm
echo "=== node voltages (dcOp) ==="
grep -E '^"(voutp|voutn|outpi|outni|vcm_sense|vbn|inp|inn)"' psf_dm/dcOp.dc
grep -E '^"I0\.(x1p|x1n|tail|vcmfb|ctail|dumm)"' psf_dm/dcOp.dc
echo "=== regions ==="
awk '/^"(I0\.M|MN2)/ {name=$1} /region/ {print name, $0}' psf_dm/dcOpInfo.info | head -20
echo "=== ids ==="
awk '/^"(I0\.M|MN2)/ {name=$1} /"ids" / {print name, $2}' psf_dm/dcOpInfo.info | head -20
