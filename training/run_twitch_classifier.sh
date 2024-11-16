# !/bin/bash

echo "Twitch GLASE embeddings"
python train_glase_dataset_embeddings.py --dataset='Twitch_ES' --mask='FULL' --d=7 --glase_steps=5
python train_glase_dataset_embeddings.py --dataset='Twitch_ES' --mask='M08' --d=7 --glase_steps=5
python train_glase_dataset_embeddings.py --dataset='Twitch_ES' --mask='M06' --d=7 --glase_steps=5
python train_glase_dataset_embeddings.py --dataset='Twitch_ES' --mask='M04' --d=7 --glase_steps=5
python train_glase_dataset_embeddings.py --dataset='Twitch_ES' --mask='M02' --d=7 --glase_steps=5

echo "Execution FULL"
python train_glase_classifier.py --dataset="Twitch_ES" --mask="FULL" --d=7
echo "Execution M08"
python train_glase_classifier.py --dataset="Twitch_ES" --mask="M08" --d=7
echo "Execution M06"
python train_glase_classifier.py --dataset="Twitch_ES" --mask="M06" --d=7
echo "Execution M04"
python train_glase_classifier.py --dataset="Twitch_ES" --mask="M04" --d=7
echo "Execution M02"
python train_glase_classifier.py --dataset="Twitch_ES" --mask="M02" --d=7

echo "Twitch E2E"
echo "Execution FULL"
python train_glase_e2e_classifier.py --dataset="Twitch_ES" --mask="FULL" --d=7 --alpha=0.5
echo "Execution M08"
python train_glase_e2e_classifier.py --dataset="Twitch_ES" --mask="M08" --d=7 --alpha=0.5
echo "Execution M06"
python train_glase_e2e_classifier.py --dataset="Twitch_ES" --mask="M06" --d=7 --alpha=0.5
echo "Execution M04"
python train_glase_e2e_classifier.py --dataset="Twitch_ES" --mask="M04" --d=7 --alpha=0.5
echo "Execution M02"
python train_glase_e2e_classifier.py --dataset="Twitch_ES" --mask="M02" --d=7 --alpha=0.5