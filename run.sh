#!/usr/bin/env bash

# This is my own work. (fs2776)
# Download data v1 and v2 -> Prepare 4 files: wav.scp, text, utt2spk. spk2utt -> MFCC feature extraction -> training and decoding

# Location for the data
datav1=./db1
datav2=./db2

. ./cmd.sh
. ./path.sh

stage=0
nj=40

. utils/parse_options.sh

set -euo pipefail

if [ $stage -le 0 ]; then
  local/download_and_untar.sh $datav1 1
  local/download_and_untar.sh $datav2 2
	
fi

if [ $stage -le 1 ]; then
  local/prepare_data.sh $datav1 ./datav1
  local/prepare_data.sh $datav2 ./datav2
fi

# For each of dataset v1 and v2, extract low resolution MFCC and high resolution MFCC.
if [ $stage -le 2 ]; then
  for x in train validation test; do 
    steps/make_mfcc.sh --nj $nj datav1/$x exp/make_mfcc/$x mfcc_v1
    steps/compute_cmvn_stats.sh datav1/$x exp/make_mfcc/$x mfcc_v1
    utils/fix_data_dir.sh datav1/$x
  done
  for x in train validation test; do 
    steps/make_mfcc.sh --nj $nj datav2/$x exp/make_mfcc_v2/$x mfcc_v2
    steps/compute_cmvn_stats.sh datav2/$x exp/make_mfcc_v2/$x mfcc_v2
    utils/fix_data_dir.sh datav2/$x
  done
  for x in train validation test; do 
    steps/make_mfcc.sh --nj $nj --mfcc_config conf/mfcc_high.conf datav1/$x exp_high/make_mfcc_v1/$x mfcc_v1_high
    steps/compute_cmvn_stats.sh datav1/$x exp_high/make_mfcc_v1/$x mfcc_v1_high
    utils/fix_data_dir.sh datav1/$x
  done
  for x in train validation test; do 
    steps/make_mfcc.sh --nj $nj --mfcc_config conf/mfcc_high.conf datav2/$x exp_high/make_mfcc_v2/$x mfcc_v2_high
    steps/compute_cmvn_stats.sh datav2/$x exp_high/make_mfcc_v2/$x mfcc_v2_high
    utils/fix_data_dir.sh datav2/$x
  done

fi

if [ $stage -le 3 ]; then
  local/run_transformer.sh
fi
