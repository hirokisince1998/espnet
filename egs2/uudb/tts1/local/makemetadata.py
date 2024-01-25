from xml.dom import minidom
from glob import glob
from os.path import join, basename
import numpy as np
import pandas as pd

utteranceFields = ["UtteranceID", "Channel", "UtteranceStartTime", "UtteranceEndTime"]
ratingFields = ["Pleasantness", "Arousal", "Dominance", "Credibility", "Interest", "Positivity"]

def ifattribute(element, name):
    anode = element.getAttributeNode(name)
    if anode is not None:
        if anode.nodeValue == "true":
            return True
    return False


def uudb_df(uudbroot):

    records = []

    UUDBSessions = join(uudbroot, "Sessions")
    sessions = sorted([basename(p) for p in glob(join(UUDBSessions, "C???"))])
    for session in sessions:
        dom = minidom.parse(join(UUDBSessions, session, "{}.xml".format(session)))
        speakerIDs = {
            "L": dom.documentElement.getAttributeNode("LSpeakerID").nodeValue,
            "R": dom.documentElement.getAttributeNode("RSpeakerID").nodeValue
        }
        sessionStartTime = float(dom.documentElement.getAttributeNode("SessionStartTime").nodeValue)
        record = {}
        for utterance in dom.getElementsByTagName("Utterance"):
            record = {fld: utterance.getAttributeNode(fld).nodeValue for fld in utteranceFields}
            record["SessionID"] = session
            record["Speaker"] = speakerIDs[record["Channel"]]
            record["SessionStartTime"] = sessionStartTime

            # ratings
            ratings = []
            for rating in utterance.getElementsByTagName("Rating"):
                nodedict = {fld: rating.getAttributeNode(fld) for fld in ratingFields}
                rdict = {fld: v.nodeValue for fld, v in nodedict.items() if v is not None}
                ratings.append(rdict)
            for axis in ratingFields:
                theratings = [int(r[axis]) for r in ratings if axis in r]
                theratings = np.array(theratings)
                record[axis] = theratings.mean()

            # PhoneticTranscription
            transcriptions = []
            numLaugh = 0
            for child in utterance.childNodes:
                if child.nodeName == "Chunk":
                    trans = child.getAttributeNode("PhoneticTranscription").nodeValue
                    if ifattribute(child, "ExpressiveInterjection") or ifattribute(child, "Filler"):
                        transcriptions.append("[")
                        transcriptions.append(trans)
                        transcriptions.append("]")
                    else:
                        transcriptions.append(trans)
                elif child.nodeName == "NonLinguisticSound":
                    if ifattribute(child, "TagBreath"):
                        transcriptions.append("{breath}")
                    elif ifattribute(child, "TagLaugh"):
                        transcriptions.append("{laugh}")
                        numLaugh += 1
                    elif ifattribute(child, "TagSigh"):
                        transcriptions.append("{sigh}")
                    elif ifattribute(child, "TagCough"):
                        transcriptions.append("{cough}")
                elif child.nodeName == "ShortPause":
                    transcriptions.append("\u3001")

            record["PhoneticTranscription"] = "".join(transcriptions)

            # numMorae
            record["numMorae"] = utterance.getElementsByTagName("Mora").length

            # numLaugh
            record["numLaugh"] = numLaugh

            records.append(record)

    uudbdf = pd.DataFrame(records, columns=records[0].keys())

    uudbdf[["startTime", "endTime"]] = uudbdf[["UtteranceStartTime", "UtteranceEndTime"]].apply(pd.to_numeric)
    uudbdf = uudbdf.assign(startTime = uudbdf.startTime - uudbdf.SessionStartTime,
                           endTime = uudbdf.endTime - uudbdf.SessionStartTime)
    uudbdf = uudbdf.drop(columns=["SessionStartTime", "UtteranceStartTime", "UtteranceEndTime"])

    return uudbdf


if __name__=="__main__":
    print(uudb_df("/home/corpus/UUDB"))
