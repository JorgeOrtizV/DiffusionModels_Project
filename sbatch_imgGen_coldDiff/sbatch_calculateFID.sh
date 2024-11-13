#!/bin/bash
source /scratch_net/spumetti/ortizj/conda/etc/profile.d/conda.sh
conda activate pytcu11
python -m pytorch_fid produced_imgs/coldDiff1 original_testImgs
