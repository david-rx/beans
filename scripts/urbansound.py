from pathlib import Path
import pandas as pd
import os
from plumbum import local


PATH_TO_CSV = "/Users/davidrobinson/Code/datasets/UrbanSound8K/metadata/UrbanSound8k.csv"
PATH_TO_AUGMENTATIONS_CSV = "/Users/davidrobinson/Code/animals/AudioAug-Diffusion/urbansound_augmentations.csv"

ONLY_AUGMENTATIONS = False

full_df = pd.read_csv(PATH_TO_CSV)
augmentation_df = pd.read_csv(PATH_TO_AUGMENTATIONS_CSV)

full_df = pd.concat([full_df, augmentation_df])

#fname, labels, mids
train_fnames = [1]
test_fnames = [2]
print(full_df.head())


for fold in range(1, 11):
    train_fnames = [i for i in range(1, 11) if i != fold] + [11]
    test_fnames = [fold]

    if ONLY_AUGMENTATIONS:
        train_fnames = [11] #only train on artificial data

    def convert(row):
        # print(row["fold"])
        if row['fold'] in train_fnames:
            split = "train"
        elif row['fold'] in test_fnames:
            split = 'test'
        else:
            split = "exclude"
            # raise NotImplementedError(f"Other folds {row['fold']} not supported")
        
        label = row["class"]

        curr_path = row["slice_file_name"]
        if "AudioAug" in curr_path:
            tgt_file = curr_path
        else:
            tgt_file = Path(f'data/urban_sound_8k/wav/fold{row["fold"]}') / (Path(str(row['slice_file_name'])).stem + '.wav')
        new_row = pd.Series({
            'path': tgt_file,
            'label': label,
            'split': split
        })
        return new_row

    # make df including train and eval splits
    df = full_df.apply(convert, axis=1)
    print(df.head())

    print(f"len {len(df)} with eval len {len(df[df['split'] == 'test'])} train len {len(df[df['split'] == 'train'])}")

    if ONLY_AUGMENTATIONS:
        df[df.split == 'train'].to_csv(f'data/urban_sound_8k/annotations_aug.train.csv')
        df[df.split == 'test'].to_csv(f'data/urban_sound_8k/annotations_aug.valid.csv')
        df[df.split == 'test'].to_csv(f'data/urban_sound_8k/annotations_aug.test.csv') # test is the same as valid
        print(f"train on augs only! test on {fold}")
        exit()
    else:
        df[df.split == 'train'].to_csv(f'data/urban_sound_8k/annotations_{fold}.train.csv')
        df[df.split == 'test'].to_csv(f'data/urban_sound_8k/annotations_{fold}.valid.csv')
        df[df.split == 'test'].to_csv(f'data/urban_sound_8k/annotations_{fold}.test.csv') # test is the same as valid