#!/usr/bin/env python3
# coding: utf-8
# make train and test sets for Paralinguistic Information Controllable TTS

import numpy as np
import pandas as pd
import os
from os.path import join
from scipy.io import wavfile
import sys
from makemetadata import uudb_df
from phonemize import mora2phoneme

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

def tokenize(text):
    phonelist = mora2phoneme(text).\
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
    uudbroot = sys.argv[1]
    outdir = sys.argv[2]
    phonemize = True

    df = uudb_df(uudbroot)
    df = df[df.Speaker.str.startswith("F")]
    df = df[df.numMorae >= 3]
    df = df[df.numLaugh == 0]
    df = df.assign(wavbn = df.apply(lambda u: f"{u.SessionID}_{u.Channel}_{u.UtteranceID}", axis=1))

    df_ = {"train": df[df.apply(lambda u: not u.wavbn in testset, axis=1)],
           "test": df[df.apply(lambda u: u.wavbn in testset, axis=1)]}

    for setn in ["train", "test"]:
        uttid = []
        spk2utt = {}
        text = []
        utt2spk = []
        wavscp = []
        for sessionID, sessiondf in df_[setn].groupby("SessionID"):
            samplerate, wav = wavfile.read(join(uudbroot, "Sessions", sessionID,
                                                f"{sessionID}.wav"))
            wav = wav.astype(np.float64) / 32768.0
            assert wav.ndim == 2
            ch = {"L": 0, "R": 1}
            for utt in sessiondf.itertuples():
                span = np.array([utt.startTime, utt.endTime])
                span = np.rint(span * samplerate).astype(int)
                range = np.arange(*span)
                uttwav = wav[range, ch[utt.Channel]]
                wavdir = join(outdir, setn, "wav")
                os.makedirs(wavdir, exist_ok=True)
                wavfn = join(wavdir, utt.wavbn + ".wav")
                wavfile.write(wavfn, samplerate, (uttwav * 32768.0).astype(np.int16))
                uttid.append(utt.wavbn)
                if utt.Speaker in spk2utt:
                    spk2utt[utt.Speaker].append(utt.wavbn)
                else:
                    spk2utt[utt.Speaker] = [utt.wavbn]
                if phonemize:
                    text.append(tokenize(utt.PhoneticTranscription))
                else:
                    text.append(utt.PhoneticTranscription)
                utt2spk.append(utt.Speaker)
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
