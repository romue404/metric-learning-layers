# MLL - Metric Learning Layers
MLL is a simple [PyTorch](https://pytorch.org/) package that includes the most common metric learning layers.
MLL only includes layers that are not dependent on negative sample mining and therefore drop in replacements for 
the final linear layer used in classification problems.
All layers aim to achieve greater inter-class variance and minimizing intra-class variance. 
Moreover, all MLL-layers can be used in conjunction with soft-targets (e.g. with [Mixup](https://arxiv.org/abs/1710.09412)).

The basis of all these layers is the scaled cosine similarity $y = xW * s$ between 
the $d$-dimensional input vectors (features) $x \in \mathbb{R}^{1 \times d}$ and the 
$c$ class weights (prototypes, embeddings) $W \in \mathbb{R}^{d \times c}$
where $||x|| = 1$ and $||W_{*, j}|| = 1 \,\, \forall j= 1\dots c$ and $s \in \mathbb{R}^+$.

## Supported Layers
We currently support the following layers:
* NormalizedLinear and ScaledNormalizedLinear
* [CosFace](https://arxiv.org/abs/1801.09414)
* [ArcFace](https://arxiv.org/abs/1801.07698)
* [AdaCos and FixedAdaCos](https://arxiv.org/abs/1905.00292)
* [DeepNCM](https://openreview.net/forum?id=rkPLZ4JPM)

You can use multiple sub-centers for all layers except for DeepNCM. If you do not specify a scale, 
MLL will use the heuristic from AdaCos $s = \sqrt{2} * log(c - 1)$.

## Install MLL
Simply run:
```
pip install metric_learning_layers
```

## Example
```py
import torch
import metric_learning_layers as mll

rnd_batch  = torch.randn(32, 128)
rnd_labels = torch.randint(low=0, high=10, size=(32, ))

arcface = mll.ArcFace(in_features=128, 
                      out_features=10, 
                      num_sub_centers=1, 
                      scale=None, # defaults to AdaCos heuristic
                      trainable_scale=False
                      )

af_out = arcface(rnd_batch, rnd_labels)  # ArcFace requires labels (used to apply the margin)
# af_out: torch.Size([32, 10])

adacos = mll.AdaCos(in_features=128, 
                    out_features=10, 
                    num_sub_centers=1 
                    )

ac_out = arcface(rnd_batch)  # AdaCos does not require labels
# af_out: torch.Size([32, 10])
```