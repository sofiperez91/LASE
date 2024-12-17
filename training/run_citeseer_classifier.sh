#!/bin/bash

python train_glase_dataset_embeddings.py --dataset='citeseer' --mask='FULL' --d=6 --glase_steps=5
python train_glase_dataset_embeddings.py --dataset='citeseer' --mask='M08' --d=6 --glase_steps=5
python train_glase_dataset_embeddings.py --dataset='citeseer' --mask='M06' --d=6 --glase_steps=5
python train_glase_dataset_embeddings.py --dataset='citeseer' --mask='M04' --d=6 --glase_steps=5
python train_glase_dataset_embeddings.py --dataset='citeseer' --mask='M02' --d=6 --glase_steps=5


echo "Execution FULL"
python train_glase_classifier.py --dataset="citeseer" --mask="FULL" --d=6
echo "Execution M08"
python train_glase_classifier.py --dataset="citeseer" --mask="M08" --d=6
echo "Execution M06"
python train_glase_classifier.py --dataset="citeseer" --mask="M06" --d=6
echo "Execution M04"
python train_glase_classifier.py --dataset="citeseer" --mask="M04" --d=6
echo "Execution M02"
python train_glase_classifier.py --dataset="citeseer" --mask="M02" --d=6

echo "Citeseer E2E"
echo "Execution E2E M08"
python train_glase_e2e_classifier.py --dataset="citeseer" --mask="M08" --d=6 --alpha=0.5
echo "Execution E2E M06"
python train_glase_e2e_classifier.py --dataset="citeseer" --mask="M06" --d=6 --alpha=0.5
echo "Execution E2E M04"
python train_glase_e2e_classifier.py --dataset="citeseer" --mask="M04" --d=6 --alpha=0.5
echo "Execution E2E M02"
python train_glase_e2e_classifier.py --dataset="citeseer" --mask="M02" --d=6 --alpha=0.5