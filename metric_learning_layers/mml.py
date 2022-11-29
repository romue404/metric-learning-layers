import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class NormalizedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, num_sub_centers: int = 1, **kwargs):
        super().__init__(in_features, out_features*num_sub_centers, False)
        self.in_features = in_features
        self.out_features = out_features
        self.num_sub_centers = num_sub_centers

    def similarities(self, data: torch.Tensor):
        normed_logits = F.normalize(data, dim=-1, p=2)
        normed_weights = F.normalize(self.weight, dim=-1, p=2)
        cos_sims = F.linear(normed_logits, normed_weights).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if self.num_sub_centers > 1:
            cos_sims = cos_sims.view(cos_sims.shape[0], -1, self.num_sub_centers)
            cos_sims, _ = cos_sims.max(-1)  # pooled similarities
        return cos_sims

    def forward(self, data: torch.Tensor, labels: Union[torch.Tensor, None] = None):
        return self.similarities(data)


def heuristic_scale(out_features: int):
    return math.sqrt(2) * math.log(out_features - 1)


class ScaledNormalizedLinear(NormalizedLinear):
    def __init__(self, *args, scale: Union[float, None] = None, trainable_scale: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        scale = scale if scale is not None else heuristic_scale(out_features)
        self.scale = nn.Parameter(torch.tensor([scale]).float(), requires_grad=trainable_scale)

    def scale_similarities(self, sims: torch.Tensor):
        return sims * self.scale

    def forward(self, data, labels=None):
        return self.scale_similarities(super().forward(data))


class ScaledNormalizedLinearSqrt(ScaledNormalizedLinear):
    def __init__(self, *args, scale=1, **kwargs):
        super().__init__(*args, scale=scale, **kwargs)
        self.d_sqrt = self.in_features**0.5

    def scale_similarities(self, sims: torch.Tensor):
        return sims * self.scale * self.d_sqrt


class CosFace(ScaledNormalizedLinear):
    def __init__(self, *args, scale=64,  margin=0.5, **kwargs):
        assert margin > 0.0, 'Margin must be > 0'
        super().__init__(*args, scale=scale, **kwargs)
        self.margin = margin

    def one_hot(self, labels: torch.Tensor):
        return F.one_hot(labels, num_classes=self.out_features) if len(labels.shape) <= 1 else labels

    def apply_margin(self, sims, labels):
        return sims - self.one_hot(labels) * self.margin

    def forward(self, data, labels):
        cos_sims = self.similarities(data)
        if self.training:
            cos_sims = self.apply_margin(cos_sims, labels)
        return self.scale_similarities(cos_sims)


class ArcFace(CosFace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_margin(self, sims, labels):
        return (sims.arccos() + self.one_hot(labels) * self.margin).cos()


class FixedAdaCos(CosFace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, scale=None, **kwargs)

    def apply_margin(self, sims, labels=None):
        return sims


class AdaCos(FixedAdaCos):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, data, labels=None):
        sims = self.similarities(data)
        scaled_sims = self.scale_similarities(sims)
        if self.training:
            device = self.scale.data.device
            f_max = scaled_sims.max()
            B_avg = (scaled_sims - f_max).exp().sum(-1).mean()
            med_angle = sims.arccos().median()
            pi4 = torch.tensor([torch.pi / 4]).to(device)
            self.scale.data = f_max + torch.log(B_avg) / (torch.min(pi4, med_angle)).cos()
        return scaled_sims


class DeepNCM(nn.Module):
    def __init__(self, features: int, classes: int, alpha: float = 0.9, scale: Union[float, None] = 10):
        super().__init__()
        self.means = nn.Parameter(torch.zeros(classes, features), requires_grad=False)
        self.running_means = nn.Parameter(torch.zeros(classes, features), requires_grad=False)
        self.alpha = alpha
        self.features = features
        self.classes = classes
        self.scale = scale if scale is not None else heuristic_scale(out_features)

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        cos_sims_scaled = (F.normalize(x, dim=-1, p=2) @ self.means.T) * self.scale
        if self.training:
            self.update_means(x, labels)
        return cos_sims_scaled

    def update_means(self, logits: torch.Tensor, labels: torch.Tensor):
        mu, update = self.compute_mean(F.normalize(logits, dim=-1, p=2), labels)
        for c, u in enumerate(update):
            if u:
                self.running_means.data[c, :] = mu[c, :]
            else:
                self.running_means.data[c, :] = self.means.data[c, :]
        self.means.data = self.alpha * self.means.data + (1 - self.alpha) * self.running_means

    @torch.inference_mode()
    def compute_mean(self, x: torch.Tensor, one_hot_labels: torch.Tensor):
        mu = (x.T @ one_hot_labels).T
        counts = one_hot_labels.sum(0)
        update = counts > 1e-7
        mu = mu / counts.view(-1, 1)
        return mu, update.squeeze()

