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
. ./db.sh || exit 1;

db_root=/home/corpus/CEJC

train_set=${spk}_tr_no_dev
train_dev=${spk}_dev
eval_set=${spk}_eval1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: local/data_prep.sh"
    local/data_prep.sh ${db_root} $spk data/$train_set
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: utils/subset_data_dir.sh"
    # make evaluation and devlopment sets
    utils/subset_data_dir.sh --first data/${train_set} 500 data/deveval
    utils/subset_data_dir.sh --first data/deveval 250 data/${eval_set}
    utils/subset_data_dir.sh --last data/deveval 250 data/${train_dev}
    n=$(( $(wc -l < data/${train_set}/wav.scp) - 500 ))
    utils/subset_data_dir.sh --last data/${train_set} ${n} data/${train_set}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
