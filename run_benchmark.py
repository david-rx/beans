import sys
from plumbum import local, FG
from plumbum.commands.processes import ProcessExecutionError

python = local['python']
local['mkdir']['-p', 'logs']()

MODELS = [
    # ('lr', 'lr', '{"C": [0.1, 1.0, 10.0]}'),
    # ('svm', 'svm', '{"C": [0.1, 1.0, 10.0]}'),
    # ('decisiontree', 'decisiontree', '{"max_depth": [None, 5, 10, 20, 30]}'),
    # ('gbdt', 'gbdt', '{"n_estimators": [10, 50, 100, 200]}'),
    # ('xgboost', 'xgboost', '{"n_estimators": [10, 50, 100, 200]}'),
    # ('resnet18', 'resnet18', ''),
    # ('clap', 'clap', ''),
    ('resnet18-pretrained', 'resnet18-pretrained', ''),
    # ('resnet50', 'resnet50', ''),
    # ('resnet50-pretrained', 'resnet50-pretrained', ''),
    # ('resnet152', 'resnet152', ''),
    # ('resnet152-pretrained', 'resnet152-pretrained', ''),
    # ('vggish', 'vggish', ''),
    # ('aves', 'aves', '')
]

TASKS = [
    # ('classification', 'urbansound8k-1'),
    # ('classification', 'urbansound8k-2'),
    # ('classification', 'urbansound8k-3'),
    # ('classification', 'urbansound8k-4'),
    # ('classification', 'urbansound8k-5'),
    # ('classification', 'urbansound8k-6'),
    # ('classification', 'urbansound8k-7'),
    # ('classification', 'urbansound8k-8'),
    # ('classification', 'urbansound8k-9'),
    # ('classification', 'urbansound8k-10'),
    # ('classification', 'watkins'),
    # ('classification', 'bats'),
    # ('classification', 'dogs'),
    ('classification', 'cbi'),
    # ('classification', 'humbugdb'),
    # ('detection', 'dcase'),
    # ('detection', 'enabirds'),
    # ('detection', 'hiceas'),
    # ('detection', 'hainan-gibbons'),
    # ('detection', 'dcase'),
    # ('detection', 'rfcx'),
    # ('classification', 'esc50'),
    # ('classification', 'speech-commands'),
]

for model_name, model_type, model_params in MODELS:
    for task, dataset in TASKS:
        print(f'Running {dataset}-{model_name}', file=sys.stderr)
        log_path = f'logs/{dataset}-{model_name}-og-plus-aug'
        try:
            if model_type in ['lr', 'svm', 'decisiontree', 'gbdt', 'xgboost']:
                python[
                    'scripts/evaluate.py',
                    '--task', task,
                    '--dataset', dataset,
                    '--model-type', model_type,
                    '--params', model_params,
                    '--log-path', log_path,
                    '--num-workers', '4'] & FG
            else:
                python[
                    'scripts/evaluate.py',
                    '--task', task,
                    '--dataset', dataset,
                    '--model-type', model_type,
                    '--batch-size', '32',
                    '--epochs', '20',
                    '--lrs', '[5e-5]', # 5e-5, 1e-4
                    '--log-path', log_path,
                    '--num-workers', '1',
                    "--model-path", ""
                    ] & FG
        except ProcessExecutionError as e:
            print(e)
