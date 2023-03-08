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
    ('resnet18-pretrained', 'resnet18-pretrained', ''),
    # ('resnet50', 'resnet50', ''),
    # ('resnet50-pretrained', 'resnet50-pretrained', ''),
    # ('resnet152', 'resnet152', ''),
    # ('resnet152-pretrained', 'resnet152-pretrained', ''),
    # ('aves', 'aves', '../aves-base-bio.pt')
    # ('vggish', 'vggish', ''),
]

TASKS = [
    ('classification', 'fsd50k'),
    ('classification', 'watkins'),
    ('classification', 'bats'),
    ('classification', 'dogs'),
    ('classification', 'cbi'),
    ('classification', 'humbugdb'),
    ('detection', 'dcase'),
    ('detection', 'enabirds'),
    ('detection', 'hiceas'),
    ('detection', 'hainan-gibbons'),
    ('detection', 'rfcx'),
    ('classification', 'esc50'),
    ('classification', 'speech-commands'),
] #not used!!

for model_name, model_type, model_params in MODELS:
    print(f'Running multitask:: - {model_name}', file=sys.stderr)
    log_path = f'logs/all-multitask-{model_name}'
    print("log path is", log_path)
    
    try:
        if model_type in ['lr', 'svm', 'decisiontree', 'gbdt', 'xgboost']:
            python[
                'scripts/evaluate_multitask.py',
                '--task', "all",
                '--datasets', "all",
                '--model-type', model_type,
                '--params', model_params,
                '--log-path', log_path,
                '--num-workers', '4'] & FG
        else:
            print("pt model, type is", model_type)
            python[
                'scripts/evaluate_multitask.py',
                '--task', "all",
                '--dataset', "all",
                '--model-type', model_type,
                '--batch-size', '8',
                '--epochs', '10',
                '--lrs', '[1e-5, 5e-5, 1e-4]',
                '--log-path', log_path,
                '--num-workers', '1',
                '--model-path', model_params] & FG
    except ProcessExecutionError as e:
        print(e)
