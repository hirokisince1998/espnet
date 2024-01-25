./tts.sh --fs 22050 --cleaner none --g2p none --tts_task gan_tts --use_sid true --train_set train --valid_set test --test_sets test --srctexts data/train/text --train_config conf/train.yaml --use_emodims true "$@"

#python3 -m espnet2.bin.tts_inference --ngpu 0 --data_path_and_name_and_type dump/raw/test/text,text,text --data_path_and_name_and_type dump/raw/test/utt2emodim,emodims,csv_float --model_file $modeldir/latest.pth --train_config $modeldir/config.yaml --output_dir /home/drobo/hiroki/vits-pic --config conf/decode.yaml --data_path_and_name_and_type test/utt2sid,sids,text_int
