#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
from os.path import join
from scipy.io import wavfile
from scipy.signal import resample
import phonemize
import sys

cejcroot = sys.argv[1] # /home/corpus/CEJC
spkid = sys.argv[2]
outdir = sys.argv[3]

# power := mean(wf^2) for 50ms
def rmscontour(waveform, fs):
    framelen = fs // 20  # 20Hz = 1/0.05s
    waveform = waveform[:(len(waveform) // framelen) * framelen]
    w2 = np.square(waveform)
    return np.sqrt(w2.reshape(-1, framelen).mean(axis=-1))


conversation_df = pd.read_csv(join(cejcroot, "metaInfo", "結合用", "conversation.csv"),dtype=str,encoding='shift_jis')
participant_df = pd.read_csv(join(cejcroot, "metaInfo", "結合用", "participant.csv"),dtype=str,encoding='shift_jis')
R_df = pd.read_table("downloads/R_log.tsv", names=["会話ID","startTime","endTime","text"])

# https://www2.ninjal.ac.jp/conversation/cejc/mediaList.html
cellphoneconvs = ["K001_011","K001_019","K004_001","K005_019a","K005_019b","K005_024","K005_033","K006_016","K010_003a","K010_003b","K010_004a","K010_004b","T006_005","T021_015"]

spkids = [spkid]

target = participant_df[
    participant_df["話者ID"].isin(spkids) &
    (participant_df["会話ID"].isin(cellphoneconvs) == False)]
spklabdict = dict(zip(target["会話ID"],target["話者ラベル"]))
sessioniddict = dict(zip(conversation_df["会話ID"],conversation_df["セッションID"]))

wavs = []
lines = []

for conversation in target["会話ID"]:
    spklab = spklabdict[conversation]
    icid = spklab[:4]
    informant = conversation[:4]
    sessiondir = join(cejcroot, "data", informant, sessioniddict[conversation])
    luu = pd.read_csv(join(sessiondir, f"{conversation}-luu.csv"), encoding='shift_jis', usecols=range(5))
    transUnit = pd.read_csv(join(sessiondir, f"{conversation}-transUnit.csv"), encoding='shift_jis', usecols=range(5))
    morphSUW = pd.read_csv(join(sessiondir, f"{conversation}-morphSUW.csv"), encoding='shift_jis')

    luu = luu[luu.speakerID == spklab]
    transUnit = transUnit[transUnit.speakerID == spklab]
    morphSUW = morphSUW[(morphSUW["話者ラベル"] == spklab)]

    luu_list = list(luu.itertuples(index=False))
    transUnit_list = list(transUnit.itertuples(index=False))
    transUnit_luuID = []
    curluu = None
    for atransunit in transUnit_list:
        if curluu is None:
            curluu = luu_list.pop(0)
        assert atransunit.startTime >= curluu.startTime and atransunit.endTime <= curluu.endTime
        transUnit_luuID.append(curluu.luuID)
        if atransunit.endTime == curluu.endTime:
            curluu = None

    transUnit["luuID"] = transUnit_luuID
        
    pron = morphSUW[["発音形出現形","転記単位の開始時刻"]].groupby("転記単位の開始時刻").agg(lambda x: ''.join(x)).reset_index()
    transUnit = pd.merge(transUnit, pron, left_on="startTime", right_on="転記単位の開始時刻").dropna()
    transUnit = transUnit[~transUnit["発音形出現形"].str.contains("◇")]
    transUnit["phoneme"] = transUnit["発音形出現形"].map(phonemize.mora2phoneme)
    transUnit = transUnit[["startTime","endTime","phoneme"]]

    conv_Rs = R_df[R_df["会話ID"] == f"{conversation}_{icid}"]
    transUnit["numR"] = transUnit.apply(lambda rec: len(conv_Rs[(conv_Rs.startTime < rec.endTime) & (conv_Rs.endTime > rec.startTime)]),axis=1)

    transUnit = transUnit[transUnit.numR == 0].drop(columns="numR")
    
    wavfn = join(sessiondir, f"{conversation}_{icid}.wav")
    samplerate, wav = wavfile.read(wavfn)
    wav = wav.astype(float)/32768.0
    # CEJC bug workaround 2023.11.30
    if conversation in ["K001_010", "K001_013"]: # 44100Hz
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        wav = resample(wav, int(len(wav)/samplerate * 16000))
        samplerate = 16000

    assert samplerate == 16000
    
    for startTime, endTime, pron in transUnit.itertuples(index=False):
        outbn = "{}_{}_{:07d}_{:07d}".format(conversation, icid, int(startTime * 1000), int(endTime * 1000))
        startsample= np.rint(startTime * samplerate).astype(int)
        endsample = np.rint(endTime * samplerate).astype(int)
        uttwav = wav[startsample:endsample]
        #wavfile.write(f"{outdir}/{outbn}.wav", samplerate, uttwav)
        wavs.append((conversation, f"{outdir}/{outbn}.wav", uttwav))
        lines.append(f"{outbn} {pron}\n")

maxrms = {}
for conversation, wavfn, uttwav in wavs:
    if not conversation in maxrms:
        maxrms[conversation] = []
    maxrms[conversation].append(np.percentile(rmscontour(uttwav, samplerate), 98))
meanmaxrms = {}
for conversation, maxrmslist in maxrms.items():
    meanmaxrms[conversation] = np.mean(maxrmslist)
maxmeanmaxrms = np.max(list(meanmaxrms.values()))

for conversation, wavfn, uttwav in wavs:
    uttwav *= maxmeanmaxrms / meanmaxrms[conversation]
    #uttwav *= 0.5 # adjustment
    wavfile.write(wavfn, samplerate, (uttwav * 32768.0).astype(np.int16))
with open(f"{outdir}/text", "w") as f:
    f.writelines(lines)
