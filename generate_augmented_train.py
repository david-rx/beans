import os
import pandas as pd

AUGMENTATIONS_FOLDER = "../audio-diffusion/augmentations_256/watkins"

def main():
    train_annotations_df = pd.read_csv("data/watkins/annotations.train.csv")
    print("start len", len(train_annotations_df))
    for folder in os.listdir(AUGMENTATIONS_FOLDER):
        folder_path = os.path.join(AUGMENTATIONS_FOLDER, folder)
        for file in os.listdir(folder_path):
            if "og" in file:
                continue
            path = os.path.join(folder_path, file)
            train_annotations_df = train_annotations_df.append({"path": path, "label": folder}, ignore_index = True)
    print("aug len", len(train_annotations_df))
    train_annotations_df.to_csv("data/watkins/augmented_annotations.train.csv")



if __name__ == "__main__":
    main()