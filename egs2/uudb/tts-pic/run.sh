./tts.sh --fs 22050 --cleaner none --g2p none --tts_task gan_tts --use_sid true --train_set train --valid_set test --test_sets test --srctexts data/train/text --train_config conf/train.yaml "$@"
