import pyopenjtalk

def conv(dir):

    with open(f"data/{dir}/text") as f:
        utts = f.readlines()

    oututts = []
    for utt in utts:
        id, text = utt.split()
        phones = pyopenjtalk.g2p(text, kana=False)
        oututts.append(f"{id} {phones}\n")
        
    with open(f"data/{dir}/text", "w") as f:
        for utt in oututts:
            f.write(utt)

if __name__ == "__main__":
    for dataset in ["dev", "deveval", "eval1", "tr_no_dev", "train"]:
        conv(dataset)
