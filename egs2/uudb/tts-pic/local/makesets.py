#!/usr/bin/env python3
# coding: utf-8
# make train and test sets for Paralinguistic Information Controllable TTS

import argparse
import numpy as np
import pandas as pd
import os
from os.path import join
from scipy.io import wavfile
import sys
from makemetadata import uudb_df
from phonemize import Phonemizer

def get_parser():
    parser = argparse.ArgumentParser(
        description="Make datasets for UUDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("uudbroot", help="directory for UUDB")
    parser.add_argument("outdir", help="output directory")
    parser.add_argument("--phonemize", action="store_true", help="convert phonetic trans to phoneme")
    parser.add_argument("--minlength", type=int, default=3, help="minimum length in mora")
    parser.add_argument("--exclude_laughter", nargs="*", help="exclude utterance with laughter from train and/or test sets (separate by spaces)")
    parser.add_argument("--delete_laughter", action="store_true", help="delete laughs from transcriptions")
    parser.add_argument("--emotion_dims", help="emotion dimensions (separate by commas)")

    return parser

testset = [
    "C002_L_107",
    "C002_L_175",
    "C004_L_126",
    "C004_R_044",
    "C005_R_152",
    "C006_L_050",
    "C007_L_137",
    "C031_L_002",
    "C033_R_134",
    "C041_L_134",
    "C041_L_259",
    "C042_R_090",
    "C043_R_064",
    "C051_L_072",
    "C051_R_118",
    "C051_R_170",
]

phonemizer = Phonemizer()

def tokenize(text):
    phonelist = phonemizer(text).\
        replace("\u3001", " sp "). \
        replace("[", " [ "). \
        replace("]", " ] "). \
        replace("{laugh}", " <laugh> "). \
        replace("{breath}", " <breath> "). \
        replace("{sigh}", " <sigh> "). \
        replace("{cough}", " <cough> "). \
        split()
    return " ".join(phonelist)    

if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    uudbroot = args.uudbroot
    outdir = args.outdir
    phonemize = args.phonemize
    minlength = args.minlength
    exclude_laughter_from = args.exclude_laughter
    delete_laughter = args.delete_laughter
    if args.emotion_dims is not None:
        emotion_dimensions = args.emotion_dims.split(',')
    else:
        emotion_dimensions = None

    df = uudb_df(uudbroot)
    df = df[df.Speaker.str.startswith("F")]
    df = df[df.numMorae >= minlength]
    df = df.assign(wavbn = df.apply(lambda u: f"{u.SessionID}{u.Speaker}_{u.UtteranceID}", axis=1)) # C001FTS_001

    df_ = {"train": df[df.apply(lambda u: not f"{u.SessionID}_{u.Channel}_{u.UtteranceID}" in testset, axis=1)],
           "test": df[df.apply(lambda u: f"{u.SessionID}_{u.Channel}_{u.UtteranceID}" in testset, axis=1)]}

    for setn in ["train", "test"]:
        uttid = []
        spk2utt = {}
        text = []
        utt2spk = []
        utt2emodim = []
        wavscp = []
        if exclude_laughter_from is not None:
            if setn in exclude_laughter_from:
                df_[setn] = df_[setn][df_[setn].numLaugh == 0]
        for sessionID, sessiondf in df_[setn].groupby("SessionID"):
            samplerate, wav = wavfile.read(join(uudbroot, "Sessions", sessionID,
                                                f"{sessionID}.wav"))
            wav = wav.astype(np.float64) / 32768.0
            assert wav.ndim == 2
            ch = {"L": 0, "R": 1}
            for utt in sessiondf.sort_values("wavbn").itertuples():
                span = np.array([utt.startTime, utt.endTime])
                span = np.rint(span * samplerate).astype(int)
                range = np.arange(*span)
                uttwav = wav[range, ch[utt.Channel]]
                wavdir = join(outdir, setn, "wav")
                os.makedirs(wavdir, exist_ok=True)
                wavfn = join(wavdir, utt.wavbn + ".wav")
                wavfile.write(wavfn, samplerate, (uttwav * 32768.0).astype(np.int16))
                uttid.append(utt.wavbn)
                kaldiSpeakerID = sessionID + utt.Speaker # C001FTS
                if kaldiSpeakerID in spk2utt:
                    spk2utt[kaldiSpeakerID].append(utt.wavbn)
                else:
                    spk2utt[kaldiSpeakerID] = [utt.wavbn]
                transcription = utt.PhoneticTranscription
                if delete_laughter:
                    transcription = transcription.replace("{laugh}", "")
                if phonemize:
                    text.append(tokenize(transcription))
                else:
                    text.append(transcription)
                utt2spk.append(kaldiSpeakerID)
                if emotion_dimensions is not None:
                    center = lambda d: ((d-4.0)) / 3.0
                    utt2emodim.append(",".join(["{:.3f}".format(center(getattr(utt, dim))) for dim in emotion_dimensions]))
                wavscp.append(wavfn)
        with open(join(outdir, setn, "spk2utt"), "w") as f:
            utts = []
            for spk in spk2utt.keys():
                utts.append(" ".join(spk2utt[spk]))
            f.writelines([f"{s} {u}\n" for s, u in zip(spk2utt.keys(), utts)])
        with open(join(outdir, setn, "text"), "w") as f:
            f.writelines([f"{u} {t}\n" for u, t in zip(uttid, text)])
        with open(join(outdir, setn, "utt2spk"), "w") as f:
            f.writelines([f"{u} {s}\n" for u, s in zip(uttid, utt2spk)])
        with open(join(outdir, setn, "wav.scp"), "w") as f:
            f.writelines([f"{u} {w}\n" for u, w in zip(uttid, wavscp)])
        if emotion_dimensions is not None:
            with open(join(outdir, setn, "utt2emodim"), "w") as f:
                f.writelines([f"{u} {w}\n" for u, w in zip(uttid, utt2emodim)])
