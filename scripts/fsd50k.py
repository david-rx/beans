from pathlib import Path
import pandas as pd

train_df = pd.read_csv("/Users/davidrobinson/Code/datasets/fsd50k/FSD50K.ground_truth/dev.csv")
test_df = pd.read_csv("/Users/davidrobinson/Code/datasets/fsd50k/FSD50K.ground_truth/eval.csv")

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
df.append(test_df.apply(convert, axis=1))

df[df.split == 'train'].to_csv('data/fsd50k/annotations.train.csv')
df[df.split == 'valid'].to_csv('data/fsd50k/annotations.valid.csv')
df[df.split == 'valid'].to_csv('data/fsd50k/annotations.test.csv') # test is the same as valid