import os
import cv2
import six
import math
import lmdb
import timm
import torch
import random
import pickle
import argparse
import numpy as np
from torch import nn
from tqdm import tqdm
import albumentations as A
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision.transforms import v2
from typing import List, Optional, Tuple, Union, no_type_check
import transformers

import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from typing import List, Optional

import cv2
import math
import random
import string
import numpy as np
from pathlib import Path
import albumentations as A
from scipy import ndimage
import scipy
import scipy.stats as ss
from scipy.interpolate import interp2d
from scipy.linalg import orth
from typing import List, Optional, Tuple, Union, no_type_check

from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from PIL import Image
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=8, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--save_root', default='/saved_model', type=str)
parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')
args = parser.parse_args()

save_root = args.save_root

import cv2
import math
import random
import numpy as np
from pathlib import Path
import albumentations as A
from typing import List, Optional, Tuple, Union, no_type_check

class CLS_DATA(Dataset):
    def __init__(self, root, images_class, types='train', img_size=384):
        self.osd = [os.path.join(root, x) for x in os.listdir(root)]
        self.lens = len(self.osd) 
        print(root, self.lens)
        self.totsr = A.Compose([
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            A.ToTensorV2(),
        ])
        self.images_class = images_class
        self.flag = (types=='train')

    def __len__(self):
        return self.lens

    def __getitem__(self, index):
        img = cv2.imread(self.osd[index])
        label = self.images_class
        img = self.totsr(image=img)['image']
        return img, label

device = torch.device("cuda",args.local_rank)
torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend='nccl')
world_size = dist.get_world_size()
nw = args.num_workers

train_dataset = torch.utils.data.ConcatDataset([CLS_DATA('Your_Real_Image_Dir', 0), CLS_DATA('Your_Fake_Image_Dir', 1)])

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,shuffle=True)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,sampler=train_sampler,)

os.makedirs(save_root, exist_ok=True)

class ViT(nn.Module):
    def __init__(self, freeze_backbone=False,):
        super(ViT, self).__init__()
        self.vit = timm.create_model('vit_large_patch16_dinov3.lvd1689m', pretrained=True, features_only=True)
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["attn.qkv"],
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        self.vit = get_peft_model(self.vit, lora_config)
        self.fc = nn.ModuleList([nn.Linear(6144, 2), nn.Linear(6144, 2), nn.Linear(6144, 2), nn.Linear(6144, 2)])
        self.sgl = BinarySupConLoss()

    def forward(self, x):
        bs = x.size(0)
        x = self.vit.model.model.forward_intermediates(x,[20,21,22,23],return_prefix_tokens=True,norm=True)[1]
        feats = [self.fc[i][0](torch.cat((x[i][0].flatten(2).mean(2).unsqueeze(1), x[i][1]), 1).reshape(bs, 6144).float()) for i in range(4)]
        preds = [self.fc[i][1:](feats[i]) for i in range(4)]
        return preds

model = ViT().to(device)
model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],output_device=args.local_rank)
model.load_state_dict(torch.load(args.load_from, map_location='cpu'))
print('loaded successfully')
lr_base = 1e-4
epochs = args.epochs
iter_per_epoch = len(train_loader)
totalstep = epochs*iter_per_epoch
warmupr = 1/epochs
warmstep = 512
lr_min = 1e-6
lr_min /= lr_base
lr_dict = {i:((((1+math.cos((i-warmstep)*math.pi/(totalstep-warmstep)))/2)+lr_min) if (i > warmstep) else (i/warmstep+lr_min)) for i in range(totalstep)}
optimizer = optim.AdamW(model.parameters(), lr=lr_base, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_dict[epoch])
ce_loss = nn.CrossEntropyLoss()
for epoch in range(args.epochs):
    model.train()
    iter_i = (epoch*iter_per_epoch)
    train_sampler.set_epoch(epoch)
    for batch_idx, (imgs, labels) in enumerate(tqdm(train_loader)):
        imgs = imgs.to(device)
        labels = labels.to(device)
        preds, feats = model(imgs)
        loss = ce_loss(preds[3], labels) + (ce_loss(preds[0], labels) + ce_loss(preds[1], labels) + ce_loss(preds[2], labels))/4.0
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(iter_i+batch_idx)
        if batch_idx%256==0:
            print(epoch, batch_idx, loss.item())
torch.save(model.state_dict(), os.path.join(args.save_root, 'e%d.pth'%epoch))
model.module.vit.save_pretrained(os.path.join(args.save_root, 'e%d_lora.pth'%epoch))
