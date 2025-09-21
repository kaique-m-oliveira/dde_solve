#!/bin/zsh
set -e  # exit on first error

python testDDE/ex_1_1_1_zennaro.py
python testDDE/ex_1_2_7.py
python testDDE/ex_1_4_1_system.py
python testDDE/ex_1_1_2_zennaro.py
python testDDE/ex_1_1_6_multiple_delays.py
python testDDE/ex1_shampine_dde23_multiple_delays.py
python testDDE/ex_ov_multi_delay.py
python testDDE/ex_ov_system.py
python testDDE/ex_ov_system_multidelay.py
