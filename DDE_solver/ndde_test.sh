#!/bin/zsh
set -e  # exit on first error

python testNDDE/ex1_paul_2_1_1.py
python testNDDE/ex2_paul_2_1_2.py
python testNDDE/ex4_paul_2_4_1.py
python testNDDE/julia_prob_dde_DDETST_H1.py
python testNDDE/NDDE_ov.py
