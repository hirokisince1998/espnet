#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=-1
stop_stage=2
spk=K001

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

db_root=/home/corpus/CEJC

dataset=${spk}
train_nodev=tr_no_dev
train_dev=dev
eval_set=eval1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    local/data_prep.sh ${db_root} $spk data/$dataset
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: utils/subset_data_dir.sh"
    # make evaluation and devlopment sets
    utils/subset_data_dir.sh --first data/${dataset} 500 data/deveval
    utils/subset_data_dir.sh --first data/deveval 250 data/${eval_set}
    utils/subset_data_dir.sh --last data/deveval 250 data/${train_dev}
    rm -rf data/deveval
    n=$(( $(wc -l < data/${dataset}/wav.scp) - 500 ))
    utils/subset_data_dir.sh --last data/${dataset} ${n} data/${train_nodev}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
