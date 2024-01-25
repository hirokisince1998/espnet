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

use_emodims=false
emotion_dims="Pleasantness,Arousal"

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

if [ -z "${UUDB}" ]; then
   log "Fill the value of 'UUDB' of db.sh"
   exit 1
fi
db_root=${UUDB}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: local/makesets.py"
    if "${use_emodims}"; then
	local/makesets.py ${db_root} data --phonemize --minlength 3 --exclude_laughter train --delete_laughter --emotion_dims ${emotion_dims}
    else
	local/makesets.py ${db_root} data --phonemize --minlength 3 --exclude_laughter train --delete_laughter
    fi
    utils/validate_data_dir.sh --no-feats data/train
fi
