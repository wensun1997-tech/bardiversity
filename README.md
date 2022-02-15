## Structured Pruning of Neural Networks with Budget Aware Regularization
Unofficial implementation of [**Structured Pruning of Neural Networks with Budget Aware Regularization**](https://arxiv.org/abs/1811.09332).

#### Installation
Run ``pip install -r requirements.txt`` to install the requirements.

#### Example
In this implementation, we pretrain, prune and finetune a ResNet network on the CIFAR-10 dataset.

Run the example experiment with the following command 

``python main.py --cfg-file configuration.yaml  --train-dir <TRAIN_DIR> --data-dir <DATA_DIR> --seed <SEED> --save-every <SAVE_EVERY> --gpu <GPU>``
