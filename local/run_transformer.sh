#!/usr/bin/env bash

# This is my own work (fs2776)
# Decode all models

. ./path.sh

echo [1/12] Running 12-class classification on dataset v1 using low resolution MFCC features...
python3 local/transformer/train.py 1 mfcc_v1 local/transformer/models/model_v1_12 12 --low_res --use_model

echo [2/12] Running 12-class classification on dataset v2 using low resolution MFCC features...
python3 local/transformer/train.py 2 mfcc_v2 local/transformer/models/model_v2_12 12 --low_res --use_model

echo [3/12] Running 21-class classification on dataset v1 using low resolution MFCC features...
python3 local/transformer/train.py 1 mfcc_v1 local/transformer/models/model_v1_21 21 --low_res --use_model

echo [4/12] Running 21-class classification on dataset v2 using low resolution MFCC features...
python3 local/transformer/train.py 2 mfcc_v2 local/transformer/models/model_v2_21 21 --low_res --use_model

echo [5/12] Running all-class classification on dataset v1 using low resolution MFCC features...
python3 local/transformer/train.py 1 mfcc_v1 local/transformer/models/model_v1_30 30 --low_res --use_model

echo [6/12] Running all-class classification on dataset v2 using low resolution MFCC features...
python3 local/transformer/train.py 2 mfcc_v2 local/transformer/models/model_v2_35 35 --low_res --use_model

echo [7/12] Running 12-class classification on dataset v1 using high resolution MFCC features...
python3 local/transformer/train.py 1 mfcc_v1_high local/transformer/models/model_v1_12_high 12 --use_model

echo [8/12] Running 12-class classification on dataset v2 using high resolution MFCC features...
python3 local/transformer/train.py 2 mfcc_v2_high local/transformer/models/model_v2_12_high 12 --use_model

echo [9/12] Running 21-class classification on dataset v1 using high resolution MFCC features...
python3 local/transformer/train.py 1 mfcc_v1_high local/transformer/models/model_v1_21_high 21 --use_model

echo [10/12] Running 21-class classification on dataset v2 using high resolution MFCC features...
python3 local/transformer/train.py 2 mfcc_v2_high local/transformer/models/model_v2_21_high 21 --use_model

echo [11/12] Running all-class classification on dataset v1 using high resolution MFCC features...
python3 local/transformer/train.py 1 mfcc_v1_high local/transformer/models/model_v1_30_high 30 --use_model

echo [12/12] Running all-class classification on dataset v2 using high resolution MFCC features...
python3 local/transformer/train.py 2 mfcc_v2_high local/transformer/models/model_v2_35_high 35 --use_model
