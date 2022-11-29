import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class NormalizedLinear(nn.Linear):
    r"""
    Normalizes the input features and class embeddings (prototypes) and computes their cosine similarity.

    Args:
        in_features (int)    : Dimensionality of the input
        out_features (int)   : Dimensionality of the output (e.g. number of classes)
        num_sub_centers (int): Number of subcenters (default=1)

    Shape:
        - Input: :math:`(*, d_{in})`
        - Output: :math:`(*, d_{out})`
    """
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


def one_hot(labels: torch.Tensor, num_classes: int):
    return F.one_hot(labels, num_classes=num_classes).float() if len(labels.shape) <= 1 else labels


class ScaledNormalizedLinear(NormalizedLinear):
    r"""
    Normalizes the input features and class embeddings (prototypes) and computes their scaled cosine similarity.

    Args:
        in_features (int)    : Dimensionality of the input
        out_features (int)   : Dimensionality of the output (e.g. number of classes)
        num_sub_centers (int): Number of subcenters (default=1)
        scale (float)        : scale to multiply the cosine similarities by

    Shape:
        - Input: :math:`(*, d_{in})`
        - Output: :math:`(*, d_{out})`
    """
    def __init__(self, in_features: int, out_features: int, num_sub_centers: int = 1,
                 scale: Union[float, None] = None, trainable_scale: bool = False, **kwargs):
        super().__init__(in_features, out_features, num_sub_centers)
        scale = scale if scale is not None else heuristic_scale(out_features)
        self.scale = nn.Parameter(torch.tensor([scale]).float(), requires_grad=trainable_scale)

    def scale_similarities(self, sims: torch.Tensor):
        return sims * self.scale

    def forward(self, data: torch.Tensor, labels: Union[torch.Tensor, None] = None):
        return self.scale_similarities(super().forward(data))


class ScaledNormalizedLinearSqrt(ScaledNormalizedLinear):
    r"""
    Normalizes the input features and class embeddings (prototypes) and computes their scaled cosine similarity.
    Subsequently, the similarities are multiplied by the square root of the input dimensionality and scaled.

    Args:
        in_features (int)    : Dimensionality of the input
        out_features (int)   : Dimensionality of the output (e.g. number of classes)
        num_sub_centers (int): Number of subcenters (default=1)
        scale (float)        : scale of the square root of the input dimensionality

    Shape:
        - Input: :math:`(*, d_{in})`
        - Output: :math:`(*, d_{out})`
    """
    def __init__(self, *args, scale=1, **kwargs):
        super().__init__(*args, scale=scale, **kwargs)
        self.d_sqrt = self.in_features**0.5

    def scale_similarities(self, sims: torch.Tensor):
        return sims * self.scale * self.d_sqrt


class CosFace(ScaledNormalizedLinear):
    r"""
    CosFace: Large Margin Cosine Loss for Deep Face Recognition
    Paper: https://arxiv.org/abs/1801.09414
    Normalizes the input features and class embeddings (prototypes) and computes their scaled cosine similarity.
    Subsequently, a cosine margin is subtracted from the similarities
    of the true labels to achieve greater inter-class variance and minimizing intra-class variance.

    Args:
        in_features (int)    : Dimensionality of the input
        out_features (int)   : Dimensionality of the output (e.g. number of classes)
        num_sub_centers (int): Number of subcenters (default=1)
        scale (float)        : Scale to multiply the cosine similarities by
        margin (float)       : Cosine margin

    Shape:
        - Input: :math:`(*, d_{in})`
        - Labels: :math:`(*)` or :math:`(*, d_{out})` either a list of int oder one_hot encoded labels
        - Output: :math:`(*, d_{out})`
    """
    def __init__(self, *args, scale: Union[float, None] = 64,  margin: float = 0.5, **kwargs):
        assert margin > 0.0, 'Margin must be > 0'
        super().__init__(*args, scale=scale, **kwargs)
        self.margin = margin

    def apply_margin(self, sims: torch.Tensor, labels: torch.Tensor):
        return sims - one_hot(labels, self.out_features) * self.margin

    def forward(self, data: torch.Tensor, labels: torch.Tensor):
        cos_sims = self.similarities(data)
        if self.training:
            cos_sims = self.apply_margin(cos_sims, labels)
        return self.scale_similarities(cos_sims)


class ArcFace(CosFace):
    r"""
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    Paper: https://arxiv.org/abs/1801.07698
    Normalizes the input features and class embeddings (prototypes) and computes their scaled cosine similarity.
    Subsequently, similarities and converted to radians and an additive angular margin is
    applied to the angular similarities of the true labels. Afterwards, the angular similarities
    are converted back to cosine similarities.

    Args:
        in_features (int)    : Dimensionality of the input
        out_features (int)   : Dimensionality of the output (e.g. number of classes)
        num_sub_centers (int): Number of subcenters (default=1)
        scale (float)        : Scale to multiply the cosine similarities by
        margin (float)       : Additive angular margin

    Shape:
        - Input: :math:`(*, d_{in})`
        - Labels: :math:`(*)` or :math:`(*, d_{out})` either a list of int oder one_hot encoded labels
        - Output: :math:`(*, d_{out})`
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_margin(self, sims: torch.Tensor, labels: torch.Tensor):
        return (sims.arccos() + one_hot(labels, self.out_features) * self.margin).cos()


class FixedAdaCos(CosFace):
    r"""
    AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations
    Paper: https://arxiv.org/abs/1905.00292
    Normalizes the input features and class embeddings (prototypes) and computes their scaled cosine similarity.
    The scale is fixed according to a heuristic that depends on
    the number of output features (see paper for more details).

    Args:
        in_features (int)    : Dimensionality of the input
        out_features (int)   : Dimensionality of the output (e.g. number of classes)
        num_sub_centers (int): Number of subcenters (default=1)

    Shape:
        - Input: :math:`(*, d_{in})`
        - Output: :math:`(*, d_{out})`
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, scale=None, trainable_scale=False, **kwargs)

    def apply_margin(self, sims: torch.Tensor, labels=None):
        return sims


class AdaCos(FixedAdaCos):
    r"""
    AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations
    Paper: https://arxiv.org/abs/1905.00292
    Normalizes the input features and class embeddings (prototypes) and computes their scaled cosine similarity.
    The scale is initialized according to a heuristic that depends on
    the number of output features (see paper for more details). The scale is updated during training.
    This implementation supports soft targets (i.e. can be used with mixup)
    as suggested in https://www.wilkinghoff.com/publications/ijcnn21_sub-cluster.pdf.

    Args:
        in_features (int)    : Dimensionality of the input
        out_features (int)   : Dimensionality of the output (e.g. number of classes)
        num_sub_centers (int): Number of subcenters (default=1)

    Shape:
        - Input: :math:`(*, d_{in})`
        - Output: :math:`(*, d_{out})`
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, data: torch.Tensor, labels=None):
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
    r"""
    DeepNCM: Deep Nearest Class Mean Classifiers
    Paper: https://openreview.net/forum?id=rkPLZ4JPM
    Normalizes the input features and class embeddings (prototypes) and computes their scaled cosine similarity.
    The prototypes are computed by an exponential moving average (EMA) of the features in the corresponding class.

    Args:
        in_features (int)    : Dimensionality of the input
        out_features (int)   : Dimensionality of the output (e.g. number of classes)
        alpha (float)        : Alpha for EMA updates
        scale (float)        : Scale to multiply the cosine similarities by

    Shape:
        - Input: :math:`(*, d_{in})`
        - Labels: :math:`(*)` or :math:`(*, d_{out})` either a list of int oder one_hot encoded labels
        - Output: :math:`(*, d_{out})`
    """
    def __init__(self, in_features: int, out_features: int, alpha: float = 0.9, scale: Union[float, None] = None):
        super().__init__()
        self.means = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.running_means = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        self.alpha = alpha
        self.features = in_features
        self.classes = out_features
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
    def compute_mean(self, x: torch.Tensor, labels: torch.Tensor):
        one_hot_labels = one_hot(labels, self.classes)
        mu = (x.T @ one_hot_labels).T
        counts = one_hot_labels.sum(0)
        update = counts > 1e-7
        mu = mu / counts.view(-1, 1)
        return mu, update.squeeze()