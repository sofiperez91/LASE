#!/bin/bash

python link_prediction_senators.py --mask_threshold=0.7 --iter=10
python link_prediction_senators.py --mask_threshold=0.5 --iter=10
python link_prediction_senators.py --mask_threshold=0.3 --iter=10
python link_prediction_senators.py --mask_threshold=0.1 --iter=10
