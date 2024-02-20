#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=22050

opts=

train_set=train
valid_set=test
test_sets=test

tts_task=gan_tts
use_sid=true
use_emodims=true
train_config=conf/train.yaml
inference_config=conf/decode.yaml
inference_model=train.total_count.ave.pth

# Input example: こ、こんにちは
# (e.g. k o sp k o N n i t i w a)

./tts.sh \
    --fs "${fs}" \
    --cleaner none \
    --g2p none \
    --tts_task "${tts_task}" \
    --use_sid "${use_sid}" \
    --use_emodims "${use_emodims}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --inference_model "${inference_model}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    ${opts} "$@"
