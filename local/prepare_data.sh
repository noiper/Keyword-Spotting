#!/usr/bin/env bash

# This is my own work. (fs2776)
# It prepares wave.scp, text, utt2spk and spk2utt files. The train validation-test-split is according to the validation_list and testing_list.

. ./path.sh

src=$1
dst=$2

[ ! -d $src ] && echo "$0: no such directory $src" && exit 1;

mkdir -p $dst || exit 1;
mkdir -p $dst/train;
mkdir -p $dst/validation;
mkdir -p $dst/test;

# Validation
awk -v "dir=$src" '{len=split($0,a,"[/|.]"); print a[2]"_"a[1]" "dir"/"a[1]"/"a[2]"."a[3]}' $src/validation_list.txt | sort >$dst/validation/wav.scp
awk '{len=split($0,a,"[/|.]");if(len==3){print a[2]"_"a[1]" "a[1]}}' $src/validation_list.txt | sort >$dst/validation/text
awk '{len=split($0,a,"[/|.]");len2=split(a[2],b,"[_]");print a[2]"_"a[1]" "b[1]}' $src/validation_list.txt | sort >$dst/validation/utt2spk
utils/utt2spk_to_spk2utt.pl<$dst/validation/utt2spk>$dst/validation/spk2utt

# Test
awk -v "dir=$src" '{len=split($0,a,"[/|.]"); print a[2]"_"a[1]" "dir"/"a[1]"/"a[2]"."a[3]}' $src/testing_list.txt | sort >$dst/test/wav.scp
awk '{len=split($0,a,"[/|.]");if(len==3){print a[2]"_"a[1]" "a[1]}}' $src/testing_list.txt | sort >$dst/test/text
awk '{len=split($0,a,"[/|.]");len2=split(a[2],b,"[_]");print a[2]"_"a[1]" "b[1]}' $src/testing_list.txt | sort >$dst/test/utt2spk
utils/utt2spk_to_spk2utt.pl<$dst/test/utt2spk>$dst/test/spk2utt

# Train
find $src | sort | egrep ".wav$" |  awk  '{len = split($0,a,"[/|.]");if (len==6){print a[4]"/"a[5]"."a[6]}}'>$src/allword_list.txt
# Create a training list that contains all data not belonging to test or validation set.
grep -Fvf  $src/testing_list.txt $src/allword_list.txt>$src/tmp.tmp
grep -Fvf  $src/validation_list.txt $src/tmp.tmp>$src/train_list.txt

awk -v "dir=$src" '{len=split($0,a,"[/|.]"); print a[2]"_"a[1]" "dir"/"a[1]"/"a[2]"."a[3]}' $src/train_list.txt | sort >$dst/train/wav.scp
awk '{split($0,a,"[/|.]");print a[2]"_"a[1]" "a[1]}' $src/train_list.txt | sort >$dst/train/text
awk '{len=split($0,a,"[/|.]");len2=split(a[2],b,"[_]");print a[2]"_"a[1]" "b[1]}' $src/train_list.txt | sort >$dst/train/utt2spk
utils/utt2spk_to_spk2utt.pl<$dst/train/utt2spk>$dst/train/spk2utt

echo "Preparing data: Success"

grep -Ff  $src/testing_list.txt $src/validation_list.txt
grep -Ff  $src/train_list.txt $src/validation_list.txt
grep -Ff  $src/testing_list.txt $src/train_list.txt

rm $src/tmp.tmp
rm $src/allword_list.txt
rm $src/train_list.txt

utils/validate_data_dir.sh --no-feats $dst/validation || exit 1
utils/validate_data_dir.sh --no-feats $dst/test || exit 1
utils/validate_data_dir.sh --no-feats $dst/train || exit 1

  


