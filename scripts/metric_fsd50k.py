import yaml
from beans.multitask import MODEL_DIR
from scripts.evaluate_multitask import load_multitask_dataset
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import wandb

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from beans.models import Aves_Metric_Learning, AvesClassifier, ResNetClassifier, VGGishClassifier, ResNetMetricLearning
from beans.datasets import ClassificationDataset, RecognitionDataset

LOG_STEP = 10


def read_datasets(path):
    with open(path) as f:
        datasets = yaml.safe_load(f)

    return {d['name']: d for d in datasets}

def read_classification_datasets(path):
    with open(path) as f:
        datasets = yaml.safe_load(f)

    return {d['name']: d for d in datasets if d["type"] == "classification"}


def train_multitask(model, task_sampler_train, epochs, lr, device, steps_per_epoch, fp16 = False):
    # wandb.init("multitask-metric-learning")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = optim.LinearLR(optimizer, start_factor=0.5, total_iters=epochs)
    
    model = model.to(device)
    model.train()
    batch_size = 16
    

    for epoch in range(epochs):
        

        task_sampler_train.reset_iterator_dict()
        running_loss = 0.0
        for i in tqdm(range(steps_per_epoch)):
            task_name, task_dataloader = task_sampler_train.pop()
            # Get the inputs and labels from the data loader
            try:
                inputs, labels = next(task_dataloader)
                inputs = inputs.to(device)
                labels = labels.to(device)
            except StopIteration:
                print("exhausted", task_name)
                continue
            

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            if fp16:
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                    loss, _ = model(inputs, labels)
            else:
                loss, _ = model(inputs, labels)

            loss.backward()
            optimizer.step()
            # Print statistics
            running_loss += loss.item()
            if i % LOG_STEP == 9:    # Print every 10 mini-batches
                # wandb.log({"epoch": epoch +1,
                #            "step": i + 1,
                #            "loss": running_loss / LOG_STEP,
                # })
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / LOG_STEP))
                running_loss = 0.0

        torch.save(model.state_dict(), f"{MODEL_DIR}multitask_metric_learning__reloaded_epoch_25_bs_128{epoch}_bs_{batch_size}")

    print('Finished multitask metric training')

def train(model, dataset_train, epochs, lr, device):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model = model.to(device)
    model.train()
    batch_size = 128
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in tqdm(enumerate(train_loader, 0)):
            # Get the inputs and labels from the data loader
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            loss, _ = model(inputs, labels)

            loss.backward()
            optimizer.step()
            # Print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # Print every 10 mini-batches
                # wandb.log({"epoch": epoch +1,
                #            "step": i + 1,
                #            "loss": running_loss,
                # })
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    torch.save(model.state_dict(), f"{MODEL_DIR}multitask_metric_learning__reloaded_epoch_14_bs_128{epochs}_bs_{batch_size}")
    print('Finished training')

def visualize(model, dataset, number_to_visualize, device):
    # Set the model to evaluation mode
    model = model.to(device)
    model.eval()

    # Sample a subset of the dataset
    indices = np.random.choice(len(dataset), number_to_visualize, replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    print("subset is", subset)

    # Embed the subset using the model
    with torch.no_grad():
        batch_size = 4
        dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)
        embeddings = []
        labels = []
        for batch in dataloader:
            input_data, label = batch
            input_data = input_data.to(device)
            _, embedding = model(input_data)
            embeddings.extend(embedding.cpu().numpy())
            labels.extend(label)

    # Apply t-SNE to the embeddings
    embeddings = np.stack(embeddings)
    print("all embeddings shape", embeddings.shape)
    tsne = TSNE(n_components=2, perplexity=30.0, n_iter=1000, random_state=0)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # Plot the t-SNE embeddings colored by label
    plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap='jet')
    plt.colorbar()
    plt.savefig(f"{model.__class__.__name__}_embedding_space.png", dpi=300)
    plt.show()


def run_multitask(model_path: str = "", resnet_model_type: str = 'resnet152', load_model = False, resnet_model = True):
    datasets = read_classification_datasets('datasets.yml')
    if resnet_model:
        model = ResNetMetricLearning(model_type=resnet_model_type, pretrained=True)
    else:
        model = Aves_Metric_Learning(model_path=model_path)
    device = torch.device("mps")
    batch_size = 64
    train_sampler = load_multitask_dataset(datasets=datasets, batch_size=batch_size, feature_type="melspectrogram" if resnet_model else "waveform")
    if load_model:
        print("loading pretrained model!")
        model.load_state_dict(torch.load(MODEL_DIR + "multitask_metric_learning__reloaded_epoch_14_bs_128"))
    print(f"batch size is {batch_size} and total steps per epoch {train_sampler.task_num_examples.sum()}")
    total_steps_in_datasets = int(train_sampler.task_num_examples.sum())
    train_multitask(model, train_sampler, epochs=25, lr=2e-5, device = device, steps_per_epoch=int(total_steps_in_datasets), fp16=False)

def main(resnet_model_type: str = 'resnet18', load_model = False):
    datasets = read_datasets('datasets.yml')
    # dataset = datasets["fsd50k"]
    dataset = datasets["cbi"]
    model = ResNetMetricLearning(model_type=resnet_model_type, pretrained=True)
    device = torch.device("mps")
    dataset_train = ClassificationDataset(
            metadata_path=dataset['train_data'],
            num_labels=dataset["num_labels"],
            labels=dataset['labels'],
            unknown_label=dataset['unknown_label'],
            sample_rate=dataset['sample_rate'],
            max_duration=dataset['max_duration'],
            feature_type="melspectrogram")
    if load_model:
        print("loading pretrained model!")
        model.load_state_dict(torch.load(MODEL_DIR + "_metric_learning_cbi_epoch_20"))
    # visualize(model, dataset_train, number_to_visualize=128 * 5, device=device)
    train(model, dataset_train, epochs=10, lr=1e-5, device=device)
    visualize(model, dataset_train, number_to_visualize=128 * 5, device=device)
    # visualize(mod)

if __name__ == "__main__":
    run_multitask(load_model=False, model_path="aves-base-bio.pt")
    # main(load_model=False)