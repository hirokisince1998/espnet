#!/usr/bin/env bash

# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

db=$1
spk=$2
org_data_dir=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <corpus_dir> <target_spk> <data_dir>"
    exit 1
fi

set -euo pipefail

# check spk existence
[ ! -e ${db}/data/${spk} ] && echo "${spk} does not exist." >&2 && exit 1;

data_dir=${org_data_dir}

# check directory existence
[ ! -e ${data_dir} ] && mkdir -p ${data_dir}

# set filenames
scp=${data_dir}/wav.scp
utt2spk=${data_dir}/utt2spk
spk2utt=${data_dir}/spk2utt
text=${data_dir}/text

# check file existence
[ -e ${scp} ] && rm ${scp}
[ -e ${utt2spk} ] && rm ${utt2spk}
[ -e ${text} ] && rm ${text}

# make scp, utt2spk, and spk2utt
local/preprocess.py $db $spk $data_dir
export data_dir
perl -ne 'my ($bn, $phn)=split; print "${bn} $ENV{data_dir}/$bn.wav\n"' $text > $scp
export spk
perl -ne 'my ($bn, $phn)=split; print "${bn} $ENV{spk}\n"' $text > $utt2spk
utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
echo "finished making wav.scp, utt2spk, spk2utt."

# check
utils/fix_data_dir.sh ${data_dir}

echo "Successfully finished data preparation."
