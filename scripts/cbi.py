import sys
import pandas as pd
from plumbum import local, FG
from pathlib import Path
import random

# sox = local['sox']
# local['mkdir']['-p', 'data/cbi/wav']()
# local['kaggle']['competitions', 'download', '-p', 'data/cbi', 'birdsong-recognition', '--force'] & FG
# local['unzip']['data/cbi/birdsong-recognition.zip', '-d', 'data/cbi/'] & FG

AUGMENT = True
DF_AUG_PATH = "/Users/davidrobinson/Code/animals/AudioAug-Diffusion/cbi_augmentations.csv"

random.seed(42)
df = pd.read_csv('data/cbi/train.csv')
all_recordist = df.recordist.unique()

random.shuffle(all_recordist)
train_recordist = set(all_recordist[:int(.6 * len(all_recordist))])
valid_recordist = set(all_recordist[int(.6 * len(all_recordist)):int(.8 * len(all_recordist))])
test_recordist = set(all_recordist[int(.8 * len(all_recordist)):])

def convert(row):
    if row['recordist'] in train_recordist:
        split = 'train'
    elif row['recordist'] in valid_recordist:
        split = 'valid'
    else:
        split = 'test'

    src_file = Path('data/cbi/train_audio') / row['ebird_code'] / row['filename']
    tgt_file = Path('data/cbi/wav') / (Path(row['filename']).stem + '.wav')
    # print(f'Converting {src_file} ...', file=sys.stderr)

    # sox[src_file, '-r', '44100', '-R', tgt_file, 'remix', '-', 'trim', '0', '10']()

    new_row = pd.Series({
        'path': tgt_file,
        'label': row['ebird_code'],
        'split': split
    })

    return new_row

df = df.apply(convert, axis=1)

df_aug = pd.read_csv(DF_AUG_PATH)
df_train = df[df.split == 'train']

if AUGMENT:
    df_train = pd.concat([df_train, df_aug])
    # df_train = df_aug

df_train.to_csv('data/cbi/annotations_aug.train.csv')
# df[df.split == 'valid'].to_csv('data/cbi/annotations.valid.csv')
# df[df.split == 'test'].to_csv('data/cbi/annotations.test.csv')
