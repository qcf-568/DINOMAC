# DINOMAC
[CVPR2026 Workshop Oral] DINOMAC: First-Place Winner Solution of the CVPR2026 Robust DeepFake Detection Challenge

The official challenge page: https://www.codabench.org/competitions/12795/#/pages-tab

<img width="1986" height="1001" alt="screenshot_2026-04-13_19-53-53" src="https://github.com/user-attachments/assets/ff2e435d-40ff-4da2-93f3-a8cd1f5b7e78" />

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
First, please modify the dataset root in [Line93](https://github.com/qcf-568/DINOMAC/blob/main/train.py#L93).
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 train.py
```

## Inference
```
CUDA_VISIBLE_DEVICES=0 python3 inference.py
```

## Contact

```
202221012612@mail.scut.edu.cn
```


## Reference

```
@inproceedings{qu2026dinomac,
  title={DINOMAC: First-Place Winner Solution of the CVPR2026 Robust DeepFake Detection Challenge},
  author={Qu, Chenfan and Jin, Lianwen and Li, Junchi and Liu, Jingjing and Yu, Bohan and Xie, Jiangwei and Liu, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2026}
}
```
