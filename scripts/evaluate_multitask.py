import ast
import argparse
import copy
import itertools
import random
import sys
import yaml
from beans.multitask import get_metric_factory, get_optimizer, save_model_dict, switch_head, switch_loss, switch_metric, TASKS
from beans.sampling import ProportionalMultiTaskSampler

from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from xgboost import XGBClassifier

from beans.metrics import Accuracy, MeanAveragePrecision
from beans.models import AvesClassifier, ResNetClassifier, VGGishClassifier
from beans.datasets import ClassificationDataset, RecognitionDataset


def read_datasets(path):
    with open(path) as f:
        datasets = yaml.safe_load(f)

    return {d['name']: d for d in datasets}


def spec2feats(spec):
    spec = torch.cat([
        spec.mean(dim=1),
        spec.std(dim=1),
        spec.min(dim=1)[0],
        spec.max(dim=1)[0]])
    return spec.numpy().reshape(-1)


def eval_sklearn_model(model_and_scaler, dataloader, num_labels, metric_factory):
    total_loss = 0.
    metric = metric_factory()
    model, scaler = model_and_scaler

    for x, y in dataloader:
        xs = [spec2feats(x[i]) for i in range(x.shape[0])]
        xs_scaled = scaler.transform(xs)
        pred = model.predict(xs_scaled)
        if isinstance(model, MultiOutputClassifier):
            pred = torch.tensor(pred)
        else:
            pred = F.one_hot(torch.tensor(pred), num_classes=num_labels)
        metric.update(pred, y)

    return total_loss, metric.get_primary_metric()


def train_sklearn_model(args, dataloader_train, dataloader_valid, num_labels, metric_factory, log_file):
    print(f'Building training data ...', file=sys.stderr)

    xs = []
    ys = []
    for x, y in dataloader_train:
        xs.extend(spec2feats(x[i]) for i in range(x.shape[0]))
        ys.extend(y[i].numpy() for i in range(y.shape[0]))

    scaler = preprocessing.StandardScaler().fit(xs)
    xs_scaled = scaler.transform(xs)
    print(f"Num. features = {xs_scaled[0].shape}, num. instances = {len(xs_scaled)}", file=sys.stderr)

    params = ast.literal_eval(args.params)
    assert(isinstance(params, dict))
    param_list = [[(k, v) for v in vs] for k, vs in params.items()]
    param_combinations = itertools.product(*param_list)

    valid_metric_best = 0.
    best_model = None

    for extra_params in param_combinations:
        extra_params = dict(extra_params)
        print(f'Fitting data (params: {extra_params})...', file=sys.stderr)

        if args.model_type == 'lr':
            model = LogisticRegression(max_iter=1_000, **extra_params)
        elif args.model_type == 'svm':
            model = SVC(**extra_params)
        elif args.model_type == 'decisiontree':
            model = DecisionTreeClassifier(**extra_params)
        elif args.model_type == 'gbdt':
            model = GradientBoostingClassifier(**extra_params)
        elif args.model_type == 'xgboost':
            model = XGBClassifier(n_jobs=4, **extra_params)

        if args.task == 'detection':
            model = MultiOutputClassifier(model)

        model.fit(xs_scaled, ys)

        _, valid_metric = eval_sklearn_model(
            model_and_scaler=(model, scaler),
            dataloader=dataloader_valid,
            num_labels=num_labels,
            metric_factory=metric_factory)

        if valid_metric > valid_metric_best:
            best_model = model
            valid_metric_best = valid_metric

        print({
            'extra_params': extra_params,
            'valid': {
                'metric': valid_metric
            }}, file=log_file)

    return (best_model, scaler), valid_metric_best


def eval_pytorch_model(model, dataloader, metric_factory, device, desc):
    model.eval()
    model_copy = model.to(device)
    total_loss = 0.
    steps = 0
    metric = metric_factory()
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=desc):
            x = x.to(device)
            y = y.to(device)

            loss, logits = model_copy(x, y)
            total_loss += loss.cpu().item()
            steps += 1
            logits = logits.to("cpu")
            y = y.to("cpu")
            metric.update(logits, y)

    total_loss /= steps

    return total_loss, metric.get_primary_metric()


def train_pytorch_model(
    args,
    task_sampler_train,
    dataloader_dict_valid,
    num_labels_dict,
    device,
    log_file,
    total_steps = 100):
    total_steps = task_sampler_train.task_num_examples.sum()
    print("total num steps is:::", total_steps)

    lrs = ast.literal_eval(args.lrs)
    assert isinstance(lrs, list)

    valid_metric_best = 0.
    best_model = None

    frozen_epochs = 1

    for lr in lrs:
        print(f"lr = {lr}" , file=log_file)

        if args.model_type.startswith('resnet'):
            pretrained = args.model_type.endswith('pretrained')
            model = ResNetClassifier(
                model_type=args.model_type,
                pretrained=pretrained,
                num_classes=1000).to(device)
        elif args.model_type == 'vggish':
            model = VGGishClassifier(
                sample_rate=16000, #WHERE IS THIS USED?
                num_classes=1000,
                multi_label=(args.task=='detection')).to(device)
        elif args.model_type == 'aves':
            model = AvesClassifier(
                model_path=args.model_path,
                num_classes=1000, #dummy num classes
                multi_label=(args.task=='detection')
                ).to(device)

        

        task_head_dict = {}
        train_metric_dict = {}

        optimizer = get_optimizer(model, task_head_dict, num_labels_dict=num_labels_dict, lr=lr)



        for epoch in range(args.epochs):
            task_sampler_train.reset_iterator_dict()
            print(f'epoch = {epoch}', file=sys.stderr)

            model.train()

            # if epoch <= frozen_epochs:
            #     for name, p in model.named_parameters():
            #         p.requires_grad = False
            #         print("p.name", name)
            #     model.linear.requires_grad = True
            #     print("linear", model.linear)

            # else:
            #     for p in model.parameters():
            #         p.requires_grad = True

                


            train_loss = 0.
            train_steps = 0

            for i in tqdm(range(total_steps)):
                
                task_name, task_dataloader = task_sampler_train.pop()

                num_labels = num_labels_dict[task_name]
                switch_head(task_name=task_name, model=model, num_labels=num_labels, task_head_dict=task_head_dict)
                switch_loss(model=model, task_name=task_name)
                # optimizer = optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=lr)
                train_metric = switch_metric(task_name=task_name, metric_dict=train_metric_dict)
                try:
                    x, y = next(task_dataloader)
                except StopIteration:
                    print("exhausted", task_name)
                    continue

        
                optimizer.zero_grad()

                x = x.to(device)
                y = y.to(device)

                loss, logits = model(x, y)

                loss.backward()

                optimizer.step()

                train_loss += loss.cpu()
                train_steps += 1

                train_metric.update(logits, y)
            
            best_model_dict = {}
            best_metric_dict = {}
            # ban_list = ["cbi", "rfcx"]
            ban_list = []

            for task_name, dataloader_valid in dataloader_dict_valid.items():
                if task_name in ban_list:
                    continue

                metric_factory = get_metric_factory(task_name)
        
                num_labels = num_labels_dict[task_name]
                switch_head(model=model, task_name=task_name, task_head_dict=task_head_dict, num_labels=num_labels)
                switch_loss(model=model, task_name=task_name)
                train_metric = switch_metric(task_name=task_name, metric_dict=train_metric_dict)
                valid_metric_best = 0.
                print(f"calling eval on task {task_name} with metric {metric_factory}")
                # classification_tasks = {t[1] for t in TASKS if t[0] == "classification"}
                # detection_tasks = {t[1] for t in TASKS if t[0] == "detection"}
                # device_for_eval = "cpu" if task_name in detection_tasks else "mps"
                valid_loss, valid_metric = eval_pytorch_model(
                    model=model,
                    dataloader=dataloader_valid,
                    metric_factory=metric_factory,
                    device=device,
                    desc='valid')

                if valid_metric > valid_metric_best:
                    valid_metric_best = valid_metric
                    best_model = copy.deepcopy(model)
                    best_model_dict[task_name] = best_model
                    best_metric_dict[task_name] = valid_metric_best

                    save_model_dict(best_model_dict, suffix=f"{model.__class__.__name__}{str(lr)}")

                print({
                    "task": task_name,
                    'epoch': epoch,
                    'train': {
                        'loss': (train_loss / train_steps).cpu().item(),
                        'metric': train_metric.get_metric(),
                    },
                    'valid': {
                        'loss': valid_loss,
                        'metric': valid_metric
                    }
                }, file=log_file)
                log_file.flush()

    return best_model_dict, valid_metric_best


def main():
    print("eval multitask.")
    datasets = read_datasets('datasets.yml')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lrs', type=str)
    parser.add_argument('--params', type=str)
    parser.add_argument('--task', choices=['classification', 'detection', 'all'])
    parser.add_argument('--model-type', choices=[
        'lr', 'svm', 'decisiontree', 'gbdt', 'xgboost',
        'resnet18', 'resnet18-pretrained',
        'resnet50', 'resnet50-pretrained',
        'resnet152', 'resnet152-pretrained',
        'vggish', 'aves'])
    parser.add_argument('--dataset', choices=["all"])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--stop-shuffle', action='store_true')
    parser.add_argument('--log-path', type=str)
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()

    torch.random.manual_seed(42)
    random.seed(42)
    if args.log_path:
        log_file = open(args.log_path, mode='w')
    else:
        log_file = sys.stderr

    # device = torch.device('cuda:0')
    device = torch.device("mps")


    if args.model_type == 'vggish':
        feature_type = 'vggish'
    elif args.model_type.startswith('resnet'):
        feature_type = 'melspectrogram'
    elif args.model_type == 'aves':
        feature_type = 'waveform'
    else:
        feature_type = 'mfcc'

    
    dataloader_dict_train = {}
    dataloader_dict_val = {}
    dataloader_dict_test = {}
    num_labels_dict = {}
    task_to_num_examples_train = {}

    # ban_list = ["esc50"]
    # ban_list = ["cbi", "rfcx"]
    ban_list = []

    for dataset_name in datasets.keys():

        if dataset_name in ban_list:
            continue


        dataset = datasets[dataset_name]
        num_labels = dataset['num_labels']

        if dataset['type'] == 'classification':
            dataset_train = ClassificationDataset(
                metadata_path=dataset['train_data'],
                num_labels=num_labels,
                labels=dataset['labels'],
                unknown_label=dataset['unknown_label'],
                sample_rate=dataset['sample_rate'],
                max_duration=dataset['max_duration'],
                feature_type=feature_type)
            dataset_valid = ClassificationDataset(
                metadata_path=dataset['valid_data'],
                num_labels=num_labels,
                labels=dataset['labels'],
                unknown_label=dataset['unknown_label'],
                sample_rate=dataset['sample_rate'],
                max_duration=dataset['max_duration'],
                feature_type=feature_type)
            dataset_test = ClassificationDataset(
                metadata_path=dataset['test_data'],
                num_labels=num_labels,
                labels=dataset['labels'],
                unknown_label=dataset['unknown_label'],
                sample_rate=dataset['sample_rate'],
                max_duration=dataset['max_duration'],
                feature_type=feature_type)

        elif dataset['type'] == 'detection':
            dataset_train = RecognitionDataset(
                metadata_path=dataset['train_data'],
                num_labels=num_labels,
                labels=dataset['labels'],
                unknown_label=dataset['unknown_label'],
                sample_rate=dataset['sample_rate'],
                max_duration=60,
                window_width=dataset['window_width'],
                window_shift=dataset['window_shift'],
                feature_type=feature_type)
            dataset_valid = RecognitionDataset(
                metadata_path=dataset['valid_data'],
                num_labels=num_labels,
                labels=dataset['labels'],
                unknown_label=dataset['unknown_label'],
                sample_rate=dataset['sample_rate'],
                max_duration=60,
                window_width=dataset['window_width'],
                window_shift=dataset['window_shift'],
                feature_type=feature_type)
            dataset_test = RecognitionDataset(
                metadata_path=dataset['test_data'],
                num_labels=num_labels,
                labels=dataset['labels'],
                unknown_label=dataset['unknown_label'],
                sample_rate=dataset['sample_rate'],
                max_duration=60,
                window_width=dataset['window_width'],
                window_shift=dataset['window_shift'],
                feature_type=feature_type)
        else:
            raise ValueError(f"Invalid dataset type: {dataset['type']}")


        dataloader_train = DataLoader(
            dataset=dataset_train,
            batch_size=args.batch_size,
            shuffle=not args.stop_shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True)
        dataloader_valid = DataLoader(
            dataset=dataset_valid,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True)
        if dataset_test is not None:
            dataloader_test = DataLoader(
                dataset=dataset_test,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True)
        else:
            dataloader_test = None
        
        task_to_num_examples_train[dataset_name] = len(dataloader_train)
        dataloader_dict_train[dataset_name] = dataloader_train
        dataloader_dict_val[dataset_name] = dataloader_valid
        dataloader_dict_test[dataset_name] = dataloader_test
        num_labels_dict[dataset_name] = num_labels

    if args.task == 'classification':
        Metric = Accuracy
    elif args.task == 'detection':
        Metric = MeanAveragePrecision

    task_sampler_train = ProportionalMultiTaskSampler(task_dict=dataloader_dict_train, rng=13,
        task_to_num_examples_dict=task_to_num_examples_train)

    if args.model_type in {'lr', 'svm', 'decisiontree', 'gbdt', 'xgboost'}:
        model_and_scaler, valid_metric_best = train_sklearn_model(
            args=args,
            dataloader_train=dataloader_train,
            dataloader_valid=dataloader_valid,
            num_labels=num_labels,
            metric_factory=Metric,
            log_file=log_file)

        if dataloader_test is not None:
            _, test_metric = eval_sklearn_model(
                model_and_scaler=model_and_scaler,
                dataloader=dataloader_test,
                num_labels=num_labels,
                metric_factory=Metric)

    else:
        best_model_dict, valid_metric_best = train_pytorch_model(
            args=args,
            task_sampler_train=task_sampler_train,
            dataloader_dict_valid=dataloader_dict_val,
            num_labels_dict=num_labels_dict,
            device=device,
            log_file=log_file)

        if dataloader_dict_test is not None:
            for task_name, model in best_model_dict.items():
                metric_factory = get_metric_factory(task_name)

                _, test_metric = eval_pytorch_model(
                    model=model,
                    dataloader=dataloader_test,
                    metric_factory=metric_factory,
                    device=device,
                    desc='test')

                print(
                    "task=", task_name, 
                    'valid_metric_best = ', valid_metric_best,
                    'test_metric = ', test_metric,
                    file=log_file)

    if args.log_path:
        log_file.close()

if __name__ == '__main__':
    main()
