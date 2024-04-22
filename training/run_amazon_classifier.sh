#!/bin/bash

echo "Execution FULL"
python train_glase_classifier.py --dataset="amazon" --mask="FULL" --d=8
echo "Execution M08"
python train_glase_classifier.py --dataset="amazon" --mask="M08" --d=8
echo "Execution M06"
python train_glase_classifier.py --dataset="amazon" --mask="M06" --d=8
echo "Execution M04"
python train_glase_classifier.py --dataset="amazon" --mask="M04" --d=8
echo "Execution M02"
python train_glase_classifier.py --dataset="amazon" --mask="M02" --d=8