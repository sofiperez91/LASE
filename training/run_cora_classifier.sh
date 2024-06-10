#!/bin/bash

# python train_glase_dataset_embeddings.py --dataset='cora' --mask='FULL' --d=6 --glase_steps=5
# python train_glase_dataset_embeddings.py --dataset='cora' --mask='M08' --d=6 --glase_steps=5
# python train_glase_dataset_embeddings.py --dataset='cora' --mask='M06' --d=6 --glase_steps=5
# python train_glase_dataset_embeddings.py --dataset='cora' --mask='M04' --d=6 --glase_steps=5
# python train_glase_dataset_embeddings.py --dataset='cora' --mask='M02' --d=6 --glase_steps=5


# echo "Execution FULL"
# python train_glase_classifier.py --dataset="cora" --mask="FULL" --d=6
# echo "Execution M08"
# python train_glase_classifier.py --dataset="cora" --mask="M08" --d=6
# echo "Execution M06"
# python train_glase_classifier.py --dataset="cora" --mask="M06" --d=6
echo "Execution M04"
python train_glase_classifier.py --dataset="cora" --mask="M04" --d=6
echo "Execution M02"
python train_glase_classifier.py --dataset="cora" --mask="M02" --d=6

# echo "CORA E2E"
# echo "Execution FULL"
# python train_glase_e2e_classifier.py --dataset="cora" --mask="FULL" --d=6
# echo "Execution M08"
# python train_glase_e2e_classifier.py --dataset="cora" --mask="M08" --d=6
# echo "Execution M06"
# python train_glase_e2e_classifier.py --dataset="cora" --mask="M06" --d=6
# echo "Execution M04"
# python train_glase_e2e_classifier.py --dataset="cora" --mask="M04" --d=6
# echo "Execution M02"
# python train_glase_e2e_classifier.py --dataset="cora" --mask="M02" --d=6