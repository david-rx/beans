from beans.metrics import Accuracy, MeanAveragePrecision
import torch
import torch.optim as optim

TASKS = [
    ('classification', 'watkins'),
    ('classification', 'fsd50k'),
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
]

MODEL_DIR = "models/"

def get_metric_factory(task_name):
    classification_tasks = {t[1] for t in TASKS if t[0] == "classification"}
    detection_tasks = {t[1] for t in TASKS if t[0] == "detection"}
    if task_name in classification_tasks:
        Metric = Accuracy
    elif task_name in detection_tasks:
        Metric = MeanAveragePrecision
    else:
        raise NotImplementedError("Task name not recognized!")
    return Metric

def switch_metric(task_name, metric_dict):
    if task_name in metric_dict:
        return metric_dict[task_name]
    else:
        metric_dict[task_name] = get_metric_factory(task_name)()
        return metric_dict[task_name]


def switch_loss(task_name, model):
    classification_tasks = {t[1] for t in TASKS if t[0] == "classification"}
    detection_tasks = {t[1] for t in TASKS if t[0] == "detection"}

    if task_name in classification_tasks:
        model.loss_func = torch.nn.CrossEntropyLoss()
    elif task_name in detection_tasks:
        model.loss_func = torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError("Task name not recognized!")

def init_model_head_for_task(model, num_labels):
    in_features = model.linear.in_features
    return torch.nn.Linear(in_features=in_features, out_features=num_labels).to("cuda")

def switch_head(task_name, model, task_head_dict, num_labels):
    """
    Switch the model's linear layer in place.
    """
    if task_name in task_head_dict:
        model_head = task_head_dict[task_name]
    else:
        model_head = init_model_head_for_task(model, num_labels)
        task_head_dict[task_name] = model_head
    model.linear = model_head

def save_model_dict(model_dict, suffix):
    for task_name, model in model_dict.items():
        torch.save(model.state_dict(), MODEL_DIR + task_name + suffix)

def load_model_for_task(model, task_name, sufix):
    model.load_state_dict(torch.load(MODEL_DIR + task_name + sufix))


def get_optimizer(model, task_head_dict, num_labels_dict):
    all_parameters = list(model.parameters())
    for task_name in TASKS:
        if task_name not in task_head_dict:
            switch_head(task_name, model, task_head_dict, num_labels_dict[task_name])
        all_parameters.append(list(task_head_dict[task_name].parameters()))
    optimizer = optim.Adam(params=all_parameters, lr=lr)
    return optimizer
