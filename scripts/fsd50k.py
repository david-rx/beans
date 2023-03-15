from pathlib import Path
import pandas as pd
import os
from plumbum import local

BASE_URL = "https://zenodo.org/record/4060432/files/"
FILE_LIST = ["FSD50K.dev_audio.z01", "FSD50K.dev_audio.z02", "FSD50K.dev_audio.z03", "FSD50K.dev_audio.z04", "FSD50K.dev_audio.z05", "FSD50K.dev_audio.zip",
        "FSD50K.doc.zip", "FSD50K.eval_audio.z01", "FSD50K.eval_audio.zip", "FSD50K.ground_truth.zip", "FSD50K.metadata.zip"]

def download_data():
    for file_name in FILE_LIST:
        download_path = os.path.join(BASE_URL, file_name)
        local['wget']['-O', download_path]['-P',]
        local["zip -s 0 FSD50K.dev_audio.zip --out dev_unsplit.zip"]
        local["zip -s 0 FSD50K.eval_audio.zip --out eval_unsplit.zip"]
        local["unzip dev_unsplit.zip -d data/fsd50k"]
        local["unzip eval_unsplit.zip -d data/fsd50k"]


download_data()

train_df = pd.read_csv("FSD50K.ground_truth/dev.csv")
test_df = pd.read_csv("FSD50K.ground_truth/eval.csv")

#fname, labels, mids
train_fnames = train_df["fname"].values
test_fnames = test_df["fname"].values

def convert(row):
    if row['fname'] in train_fnames:
        split = row["split"]
    elif row['fname'] in test_fnames:
        split = 'test'
    
    label = row["labels"].split(",")[0] # leaf

    tgt_file = Path('data/fsd50k/wav') / (Path(str(row['fname'])).stem + '.wav')
    new_row = pd.Series({
        'path': tgt_file,
        'label': label,
        'split': split
    })
    return new_row

# make df including train and eval splits
df = train_df.apply(convert, axis=1)
df = df.append(test_df.apply(convert, axis=1))

print(f"len {len(df)} with eval len {len(df[df.split == 'test'])}")

df[df.split == 'train'].to_csv('data/fsd50k/annotations.train.csv')
df[df.split == 'test'].to_csv('data/fsd50k/annotations.valid.csv')
df[df.split == 'test'].to_csv('data/fsd50k/annotations.test.csv') # test is the same as valid