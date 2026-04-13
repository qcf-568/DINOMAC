# DINOMAC
[CVPR2026 Workshop Oral] First-Place Winner Solution of the CVPR2026 Robust DeepFake Detection Challenge

---

# DINOMAC
Paper title: First-Place Winner Solution of the CVPR2026 Robust DeepFake Detection Challenge



## Enviroment
Python 3.10.12
```
pip install -r requirements.txt
```
This simplified version works in most cases:
```
pip install -U torch torchvision timm lmdb
```

## Train
First, please modify the dataset root in [Line94](https://github.com/qcf-568/DINOMAC/blob/main/train.py#L94).
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 train.py
```

## Inference
CUDA_VISIBLE_DEVICES=0 python3 inference.py
