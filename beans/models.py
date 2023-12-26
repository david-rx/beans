import argparse
from beans.clap.loss import ClipLoss
import torch
import torch.nn as nn
import torchvision
from transformers import AutoProcessor, ClapModel, ClapAudioModelWithProjection, ClapProcessor
import re

KEYS_TO_MODIFY_MAPPING = {
    "text_branch": "text_model",
    "audio_branch": "audio_model.audio_encoder",
    "attn": "attention.self",
    "self.proj": "output.dense",
    "attention.self_mask": "attn_mask",
    "mlp.fc1": "intermediate.dense",
    "mlp.fc2": "output.dense",
    "norm1": "layernorm_before",
    "norm2": "layernorm_after",
    "bn0": "batch_norm",
}

class ResNetClassifier(nn.Module):
    def __init__(self, model_type, pretrained=False, num_classes=None, multi_label=False):
        super().__init__()

        if model_type.startswith('resnet50'):
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            self.resnet = torchvision.models.resnet50(weights=weights if pretrained else None)
        elif model_type.startswith('resnet152'):
            weights = torchvision.models.ResNet152_Weights.DEFAULT
            self.resnet = torchvision.models.resnet152(weights=weights if pretrained else None)
        elif model_type.startswith('resnet18'):
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            self.resnet = torchvision.models.resnet18(weights=weights if pretrained else None)
        else:
            assert False

        self.linear = nn.Linear(in_features=1000, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = x.unsqueeze(1)      # (B, F, L) -> (B, 1, F, L)
        x = x.repeat(1, 3, 1, 1)    # -> (B, 3, F, L)
        x /= x.max()            # normalize to [0, 1]
        # x = self.transform(x)

        x = self.resnet(x)
        logits = self.linear(x)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits


class VGGishClassifier(nn.Module):
    def __init__(self, sample_rate, num_classes=None, multi_label=False):
        super().__init__()

        self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.vggish.postprocess = False
        self.vggish.preprocess = False

        self.linear = nn.Linear(in_features=128, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

        self.sample_rate = sample_rate

    def forward(self, x, y=None):
        batch_size = x.shape[0]
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        out = self.vggish(x)
        out = out.reshape(batch_size, -1, out.shape[1])
        outs = out.mean(dim=1)
        logits = self.linear(outs)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits

class AvesClassifier(nn.Module):
    def __init__(self, model_path, num_classes, embeddings_dim=768, multi_label=False):

        super().__init__()

        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        self.model = models[0]
        self.model.feature_extractor.requires_grad_(False)
        self.linear = nn.Linear(in_features=embeddings_dim, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        out = self.model.extract_features(x)[0]
        out = out.mean(dim=1)             # mean pooling
        logits = self.linear(out)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits
    

class AvesEncoder(nn.Module):
    def __init__(self, model_path, num_classes, embeddings_dim=768, multi_label=False):

        super().__init__()

        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        self.model = models[0]
        self.model.feature_extractor.requires_grad_(False)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        out = self.model.extract_features(x)[0]
        out = out.mean(dim=1)             # mean pooling

        loss = None
        if y is not None:
            loss = self.loss_func(out, y)

        return loss, out
    
class Aves_Metric_Learning(nn.Module):
    def __init__(self, model_path, loss: str = "ms_loss", use_miner: bool = True, miner_margin=0.2, type_of_triplets="all") -> None:
        super().__init__()
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        self.model = models[0]
        self.model.feature_extractor.requires_grad_(False)
        if loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(alpha=1, beta=60, base=0.5) # 1,2,3; 40,50,60
        self.use_miner = use_miner
        self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)

    def forward(self, X, Y = None):
        encodings = self.model.extract_features(X)[0]
        encodings = encodings.mean(dim=1)
        if Y != None:
            hard_pairs = self.miner(encodings, Y)
            loss = self.loss(encodings, Y, hard_pairs)
        return loss, encodings

class ResNetMetricLearning(nn.Module):
    def __init__(self, model_type, pretrained=False, loss: str ="ms_loss", use_miner: bool = True, miner_margin=0.2, type_of_triplets="all"):
        super().__init__()

        if model_type.startswith('resnet50'):
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            self.resnet = torchvision.models.resnet50(weights=weights if pretrained else None)
        elif model_type.startswith('resnet152'):
            weights = torchvision.models.ResNet152_Weights.DEFAULT
            self.resnet = torchvision.models.resnet152(weights=weights if pretrained else None)
        elif model_type.startswith('resnet18'):
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            self.resnet = torchvision.models.resnet18(weights=weights if pretrained else None)
        else:
            assert False

        if loss == "ms_loss":
            self.loss = losses.MultiSimilarityLoss(alpha=1, beta=60, base=0.5) # 1,2,3; 40,50,60
        self.use_miner = use_miner
        self.miner = miners.TripletMarginMiner(margin=miner_margin, type_of_triplets=type_of_triplets)
        
    def forward(self, x, y=None):
        x = x.unsqueeze(1)      # (B, F, L) -> (B, 1, F, L)
        x = x.repeat(1, 3, 1, 1)    # -> (B, 3, F, L)
        x /= x.max()            # normalize to [0, 1]
        # x = self.transform(x)
        loss = None
        encodings = self.resnet(x)
        if y != None:
            hard_pairs = self.miner(encodings, y)
            # print(f"num hard pairs is {len(hard_pairs[0])}")
            if len(hard_pairs[0]) < 2:
                print(f"only {len(hard_pairs[0])} hard pairs")
            loss = self.loss(encodings, y, hard_pairs)

        return loss, encodings
    
class CLAPClassifier(nn.Module):
    def __init__(self, model_path, num_classes, multi_label = False, load_animals=False) -> None:
        super().__init__()
        self.clap = ClapAudioModelWithProjection.from_pretrained(model_path, projection_dim=num_classes,
                                                                ignore_mismatched_sizes=True)
        if load_animals:
            checkpoint = torch.load("./v2.1.pt", map_location="cpu")
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            state_dict = rename_state_dict(state_dict, exclude_text = True)
            self.clap.audio_model.load_state_dict(state_dict, strict=False)

        self.processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.multi_label = multi_label

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = [s.cpu().numpy() for s in x]
        inputs = self.processor(audios=x, return_tensors="pt", sampling_rate=48000, padding=True).to("mps")
        out = self.clap(**inputs).audio_embeds
        loss = self.loss_func(out, y)
        
        return loss, out
    
class CLAPContrastiveClassifier(nn.Module):
    def __init__(self, model_path, labels, multi_label = False, use_contrastive_loss = False) -> None:
        super().__init__()
        self.clap = ClapModel.from_pretrained(model_path)
        checkpoint = torch.load("./BioLingual.pt", map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        state_dict = rename_state_dict(state_dict)
        self.clap.load_state_dict(state_dict, strict=False)
        self.processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.clap.audio_model.train()
        # for param in self.clap.text_model.parameters():
        #     param.requires_grad = False
        # for param in self.clap.text_projection.parameters():
        #     param.requires_grad = False


        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
        self.use_contrastive_loss = use_contrastive_loss
        self.liner_similarity = nn.Linear(1024, 1)
        # self.loss_func = ClipLoss()
        # self.noncontrastive_loss = nn.CrossEntropyLoss()
        if self.use_contrastive_loss:
            self.loss_func = losses.MultiSimilarityLoss(alpha=1, beta=60, base=0.5) # 1,2,3; 40,50,60
            self.use_miner = True
            self.miner = miners.TripletMarginMiner(margin=0.0, type_of_triplets="all")

        self.labels = labels
        self.contrastive_percent = 0.25

    def forward(self, x, y=None, training = True):
        x = [s.cpu().numpy() for s in x]
        if training and self.use_contrastive_loss: # train-time, contrastive learning within the batch
            batch_labels = [self.labels[index] for index in y]
            inputs = self.processor(audios=x, text=batch_labels, return_tensors="pt", sampling_rate=48000, padding=True).to("mps")
        else: #test-time
            inputs = self.processor(audios=x, text=self.labels, return_tensors="pt", sampling_rate=48000, padding=True).to("mps")
        clap_output = self.clap(**inputs, return_loss=True)
        out = clap_output.logits_per_audio
        if self.use_contrastive_loss:
            text_embeds = clap_output.text_embeds
            audio_embeds = clap_output.audio_embeds
            embeds = torch.cat([text_embeds, audio_embeds])
            labels = torch.cat([y, y])
            hard_pairs = self.miner(embeds.cpu(), labels.cpu())
            loss = self.loss_func(embeds, labels, hard_pairs)
            # loss = clap_output.loss
        else:
            loss = self.noncontrastive_mm_loss(audio_embeds=clap_output.audio_embeds, text_embeds=clap_output.text_embeds, logit_scale_audio=self.clap.logit_scale_a.exp(), label_indices=y)
            # if not (torch.isnan(clap_output.loss).any() or torch.isinf(clap_output.loss).any()):
            #     print("clap output loss", clap_output.loss)
            #     loss = (1 - self.contrastive_percent) * loss + self.contrastive_percent * clap_output.loss
            # loss, out = self.noncontrastive_mm_linear_loss(audio_embeds=clap_output.audio_embeds, text_embeds=clap_output.text_embeds, logit_scale_audio=self.clap.logit_scale_a.exp(), label_indices=y)
        return loss, out
    
    def noncontrastive_mm_loss(self, audio_embeds, text_embeds, logit_scale_audio, label_indices):
        # logits_per_text = torch.matmul(text_embeds, audio_embeds.t()) * logit_scale_text
        logits_per_audio = torch.matmul(audio_embeds, text_embeds.t()) * logit_scale_audio #only optimize on text prediction
        print(logits_per_audio)
        return self.loss_func(logits_per_audio, label_indices)
    
    def noncontrastive_mm_linear_loss(self, audio_embeds, text_embeds, logit_scale_audio, label_indices):
        text_embeds_by_batch = text_embeds.repeat(audio_embeds.shape[0], 1).view(-1, 512)
        audio_embeds_by_labels = audio_embeds.repeat(len(self.labels), 1).view(-1, 512)
        mm_embeds = torch.cat((audio_embeds_by_labels, text_embeds_by_batch), dim=-1)
        logits_per_audio = self.liner_similarity(mm_embeds)
        logits_per_audio = logits_per_audio.reshape(audio_embeds.shape[0], len(self.labels))
        return self.loss_func(logits_per_audio, label_indices), logits_per_audio
    
def rename_state_dict(state_dict, exclude_text = False):
    state_dict = {(k.replace("module.", "", 1) if k.startswith("module.") else k): v for k, v in state_dict.items()}

    model_state_dict = {}

    sequential_layers_pattern = r".*sequential.(\d+).*"
    text_projection_pattern = r".*_projection.(\d+).*"

    for key, value in state_dict.items():
        if exclude_text and "text_branch" in key:
            continue
        # check if any key needs to be modified
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        if re.match(sequential_layers_pattern, key):
            # replace sequential layers with list
            sequential_layer = re.match(sequential_layers_pattern, key).group(1)

            key = key.replace(f"sequential.{sequential_layer}.", f"layers.{int(sequential_layer)//3}.linear.")
        elif re.match(text_projection_pattern, key):
            projecton_layer = int(re.match(text_projection_pattern, key).group(1))

            # Because in CLAP they use `nn.Sequential`...
            transformers_projection_layer = 1 if projecton_layer == 0 else 2

            key = key.replace(f"_projection.{projecton_layer}.", f"_projection.linear{transformers_projection_layer}.")

        if "audio" and "qkv" in key:
            # split qkv into query key and value
            mixed_qkv = value
            qkv_dim = mixed_qkv.size(0) // 3

            query_layer = mixed_qkv[:qkv_dim]
            key_layer = mixed_qkv[qkv_dim : qkv_dim * 2]
            value_layer = mixed_qkv[qkv_dim * 2 :]

            model_state_dict[key.replace("qkv", "query")] = query_layer
            model_state_dict[key.replace("qkv", "key")] = key_layer
            model_state_dict[key.replace("qkv", "value")] = value_layer
        else:
            model_state_dict[key] = value

    return model_state_dict

class CLAPZeroShotClassifier(nn.Module):

    def __init__(self, model_path, labels, multi_label=False) -> None:
        super().__init__()
        self.clap = ClapModel.from_pretrained("davidrrobinson/BioLingual")
        checkpoint = torch.load("./biolingual-1.5.3e20.pt", map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        state_dict = rename_state_dict(state_dict)
        self.clap.load_state_dict(state_dict, strict=False)
        # self.clap.push_to_hub("BioLingual")
        self.processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
        # self.processor.push_to_hub("BioLingual")
        self.loss_func = nn.CrossEntropyLoss()
        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        self.labels = labels
        self.multi_label = multi_label

    def forward(self, x, y=None):
        x = [s.cpu().numpy() for s in x]
        inputs = self.processor(audios=x, text=self.labels, return_tensors="pt", sampling_rate=48000, padding=True).to("mps")
        if self.multi_label:
            out = self.clap(**inputs).logits_per_audio
        else:
            out = self.clap(**inputs).logits_per_audio

        loss = self.loss_func(out, y)
        return loss, out
    
    def predict_multilabel(self, inputs):
        clap_output = self.clap(**inputs, return_loss=True)
        audio_embeds = clap_output.audio_embeds
        text_embeds = clap_output.text_embeds
        logit_scale_audio=self.clap.logit_scale_a.exp()
        logits_per_audio = torch.matmul(audio_embeds, text_embeds.t()) * logit_scale_audio
        print(logits_per_audio)
