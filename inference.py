import os
import cv2
import six
import math
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
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from PIL import Image
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, default='publictest_data_final', help='input image dir')
args = parser.parse_args()

os.makedirs('submission', exist_ok=True)

import cv2
import math
import random
import numpy as np
from pathlib import Path
import albumentations as A
from typing import List, Optional, Tuple, Union, no_type_check

class CLS_DATA(Dataset):
    def __init__(self, root):
        self.root = root
        self.totsr656 = A.Compose([
            A.Resize(height=656, width=656, interpolation=cv2.INTER_LINEAR),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            A.ToTensorV2(),
        ])

    def __len__(self):
        return len(os.listdir(self.root))

    def __getitem__(self, index):
        img = cv2.cvtColor(cv2.imread(os.path.join(self.root, '%04d.png'%(index))), cv2.COLOR_BGR2RGB)
        return self.totsr656(image=img)['image']

test_dataset = CLS_DATA(args.img_dir)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1)

class ViT(nn.Module):
    def __init__(self, freeze_backbone=False, weights=[0,0,0,1]):
        super(ViT, self).__init__()
        self.vit = timm.create_model('vit_large_patch16_dinov3.lvd1689m', pretrained=True, features_only=True)
        self.fc = nn.ModuleList([nn.Sequential(nn.Linear(6144,1024), nn.Dropout(0.2), nn.Linear(1024, 2)), nn.Sequential(nn.Linear(6144,1024), nn.Dropout(0.2), nn.Linear(1024, 2)), nn.Sequential(nn.Linear(6144,1024), nn.Dropout(0.2), nn.Linear(1024, 2)), nn.Sequential(nn.Linear(6144,1024), nn.Dropout(0.2), nn.Linear(1024, 2))])
        assert np.sum(weights)==1
        self.weights = weights

    def forward(self, x):
        bs = x.size(0)
        x = self.vit.model.model.forward_intermediates(x,[20,21,22,23],return_prefix_tokens=True,norm=True)[1]
        preds = np.array([(F.softmax(self.fc[i](torch.cat((x[i][0].flatten(2).mean(2).unsqueeze(1), x[i][1]), 1).reshape(bs, 6144).float()),dim=1)[:,1]*self.weights[i]).cpu().numpy().mean(0) for i in range(4)]).sum()
        return preds

  model = ViT()
model.vit = PeftModel.from_pretrained(model1.vit, 'your_weights.pth')
model.fc.load_state_dict(torch.load('your_lora_weights.pth'))
model.eval()
model = model.cuda()
f = open('temp.txt', 'w')
with torch.no_grad():
    for batch_idx, (imgs) in enumerate(tqdm(test_loader)):
        imgs = imgs.cuda()
        preds = model(imgs)
        f.write('%.8f\n'%preds)
f.close()

def create_submission(content: str, zip_name: str) -> str:
    zip_path = f'{zip_name}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('submission.txt', content)
    return zip_path

with open('temp.txt') as f:
    content = f.read()
    zip_file = create_submission(content, 'submission/prediction')

os.remove('temp.txt')
