# Transfer-and-Fusion: Integrated Link Prediction across Knowledge Graphs


## Quick Start

### Installation
Install PyTorch following the instructions on the [PyTorch](https://pytorch.org/).
Our code is written in Python3.

- pytorch=1.11.0
- pyg=2.0.4

```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg
```


### Dataset
```
unzip dataset.zip
```

It will generate two dataset folders in the ./data directory. In our experiments, the datasets used are: `DBP-FB` and `WIKI-YAGO`.
In each dataset, there are two KGs and shared entities between them.

### Training and evaluation
```
python main.py -dataset <dataset_name> -gpu <device_id> 
```
In our experiments, the datasets used are: `DBP-FB` or `WIKI-YAGO`. 


