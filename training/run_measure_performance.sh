#!/bin/bash

python measure_performance.py --model='svd' --dataset='sbm3_unbalanced_positive_v2'
python measure_performance.py --model='LASE_ER05' --dataset='sbm3_unbalanced_positive_v2'
python measure_performance.py --model='LASE_WS03' --dataset='sbm3_unbalanced_positive_v2'
python measure_performance.py --model='LASE_BB03' --dataset='sbm3_unbalanced_positive_v2'
python measure_performance.py --model='LASE_FULL' --dataset='sbm3_unbalanced_positive_v2'
python measure_performance.py --model='cgd' --dataset='sbm3_unbalanced_positive_v2'


python measure_performance.py --model='svd' --dataset='sbm10_unbalanced_positive'
python measure_performance.py --model='LASE_ER05' --dataset='sbm10_unbalanced_positive'
python measure_performance.py --model='LASE_WS03' --dataset='sbm10_unbalanced_positive'
python measure_performance.py --model='LASE_BB03' --dataset='sbm10_unbalanced_positive'
python measure_performance.py --model='LASE_FULL' --dataset='sbm10_unbalanced_positive'
python measure_performance.py --model='cgd' --dataset='sbm10_unbalanced_positive'