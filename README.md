# ELMloss
This repository is the official PyTorch implementation ''Enlarged Large Margin Loss For Imbalanced Classification'' [[paper]](https://arxiv.org/abs/2306.09132) 

## Introduction
LDAM loss, which minimizes a margin-based generalization bound, is widely utilized for class-imbalanced image classification. Although, by using LDAM loss, it is possible to obtain large margins for the minority classes and small margins for the majority classes, the relevance to a large margin, which is included in the original softmax cross entropy loss, is not be clarified yet. In this study, we reconvert the formula of LDAM loss using the concept of the large margin softmax cross entropy loss based on the softplus function and confirm that LDAM loss includes a wider large margin than softmax cross entropy loss. Furthermore, we propose a novel Enlarged Large Margin (ELM) loss, which can further widen the large margin of LDAM loss. ELM loss utilizes the large margin for the maximum logit of the incorrect class in addition to the basic margin used in LDAM loss. Through experiments conducted on imbalanced CIFAR datasets and large-scale datasets with long-tailed distribution, we confirmed that classification accuracy was much improved compared with LDAM loss and conventional losses for imbalanced classification.
<br />
<br />

## How to use
1. Please refer to LDAM's Github page [[LDAM-DRW]](https://github.com/kaidic/LDAM-DRW)
2. Replace the loss function with ELM loss from "loss.py"

## Citation
```
@article{kato2023enlarged,
  title={Enlarged Large Margin Loss for Imbalanced Classification},
  author={Kato, Sota and Hotta, Kazuhiro},
  journal={arXiv preprint arXiv:2306.09132},
  year={2023}
}
