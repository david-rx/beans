import yaml
import pandas as pd
import json

def read_datasets(path):
    with open(path) as f:
        datasets = yaml.safe_load(f)

    return {d['name']: d for d in datasets}

def build_master_csv(datasets_path):
    datasets = read_datasets(datasets_path)
    master_csv = pd.DataFrame.from_dict({"path": [], "label": [], "task": []})
    for dataset_name, dataset in datasets.items():
        if dataset["train_data"].endswith("csv"):
            train_csv = pd.read_csv(dataset["train_data"])
            task_list = [dataset_name for i in range(len(train_csv))]
            train_csv["task"] = task_list
            master_csv = master_csv.append(train_csv)
        else:
            with open(dataset["train_data"]) as f:
                for line in f:
                    data = json.loads(line)
                    wav_path = data['path']
                    length = data['length']
                    master_csv = master_csv.append({"path": wav_path, "label": "", "task": dataset_name}, ignore_index = True)
    master_csv.to_csv("master_train.csv")
                

if __name__ == "__main__":
    build_master_csv("datasets.yml")