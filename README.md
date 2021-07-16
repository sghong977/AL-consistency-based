# Deep Active Learning with Consistency-based Regularization
Arxiv preprint
https://arxiv.org/abs/2011.02666
@misc{hong2020deep,
      title={Deep Active Learning with Augmentation-based Consistency Estimation}, 
      author={SeulGi Hong and Heonjin Ha and Junmo Kim and Min-Kook Choi},
      year={2020},
      eprint={2011.02666},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

## Deep Active Learning
Python implementations of the following active learning algorithms:

- Random Sampling
- Cutout Sampling
- CutMix Sampling
- Entropy Sampling
- Margin Sampling

### Available Dataset
- CIFAR10
- FashionMNIST

### Docker 
- nvcr.io/nvidia/pytorch:19.07-py3
- pip install torchvision==0.2.1

### Prerequisites 
- pytorch, matplotlib, scikit-learn, pandas
- torchvision      0.2.1

### Commandline Arguments
- round : # of learning cycle
- epoch : # of epoch per round
- initnum : initial data before applying AL strategy
- pick : # of data to be picked per round (notated as K)

- train_al : Cutout loss (Consistency-based Loss)
- train_cm : CutMix loss (Consistency-based Loss)

- SEED : torch.manual_seed(SEED)
- drop : epoch that apply learning rate drop (default 160)
- dataset : choose dataset among available ones
- pretrained_path : path of round0 network (for model initialization)

- ALtype : AL strategy. Random, CutoutSampling, ...
- alpha : cutout scaling
