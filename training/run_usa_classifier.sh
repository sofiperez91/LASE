# !/bin/bash

# echo "USA Airports GLASE embeddings"
# python train_glase_dataset_embeddings.py --dataset='USA' --mask='FULL' --d=4 --glase_steps=5
# python train_glase_dataset_embeddings.py --dataset='USA' --mask='M08' --d=4 --glase_steps=5
# python train_glase_dataset_embeddings.py --dataset='USA' --mask='M06' --d=4 --glase_steps=5
# python train_glase_dataset_embeddings.py --dataset='USA' --mask='M04' --d=4 --glase_steps=5
# python train_glase_dataset_embeddings.py --dataset='USA' --mask='M02' --d=4 --glase_steps=5

# echo "Execution FULL"
# python train_glase_classifier.py --dataset="USA" --mask="FULL" --d=4
# echo "Execution M08"
# python train_glase_classifier.py --dataset="USA" --mask="M08" --d=4
# echo "Execution M06"
# python train_glase_classifier.py --dataset="USA" --mask="M06" --d=4
# echo "Execution M04"
# python train_glase_classifier.py --dataset="USA" --mask="M04" --d=4
# echo "Execution M02"
# python train_glase_classifier.py --dataset="USA" --mask="M02" --d=4

echo "USA E2E"
echo "Execution FULL"
python train_glase_e2e_classifier.py --dataset="USA" --mask="FULL" --d=4 --alpha=0.5
echo "Execution M08"
python train_glase_e2e_classifier.py --dataset="USA" --mask="M08" --d=4 --alpha=0.5
echo "Execution M06"
python train_glase_e2e_classifier.py --dataset="USA" --mask="M06" --d=4 --alpha=0.5
echo "Execution M04"
python train_glase_e2e_classifier.py --dataset="USA" --mask="M04" --d=4 --alpha=0.5
echo "Execution M02"
python train_glase_e2e_classifier.py --dataset="USA" --mask="M02" --d=4 --alpha=0.5