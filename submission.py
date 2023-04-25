from glob import glob
from sklearn.model_selection import GroupKFold
import cv2
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import sklearn
from efficientnet_pytorch import EfficientNet

SEED = 42
CHECKPOINT_PATH = '/home/chloe/Chloe'
DATA_ROOT_PATH = '/home/chloe/Siting/ALASKA2'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

# Net
def get_net():
    net = EfficientNet.from_pretrained('efficientnet-b2')
    net._fc = nn.Linear(in_features=1408, out_features=4, bias=True)
    return net

net = get_net().cuda()

# Valid Augs
def get_valid_transforms():
    return A.Compose([
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

# Inference
checkpoint = torch.load(CHECKPOINT_PATH)
net.load_state_dict(checkpoint['model_state_dict']);
net.eval();
checkpoint.keys();

class DatasetSubmissionRetriever(Dataset):
    def __init__(self, image_names, transforms = None):
        super().__init__()
        self.image_names = image_names
        self.transformer = transforms
    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        image = cv2.imread(f'{DATA_ROOT_PATH}/Test/{image_name}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
        return image_name, image

dataset = DatasetSubmissionRetriever(
    image_names = np.array([path.split('/')[-1] for path in glob('/home/chloe/Siting/ALASKA2/Test/*jpg')]),
    transforms = get_valid_transforms(),
)

data_loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    drop_last=False,
)

result = {'Id': [], 'Label': []}
for step, (image_names, images) in enumerate(data_loader):
    print(step, end='\r')
    
    y_pred = net(images.cuda())
    y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:,0]
    
    result['Id'].extend(image_names)
    result['Label'].extend(y_pred)

# Submission
submission = pd.DataFrame(result)
submission.sort_values(by='Id', inplace=True)
submission.reset_index(drop=True, inplace=True)
submission.to_csv('submission_b2.csv', index=False)
submission.head()

sub_stack = pd.read_csv('')
sub_stack.sort_values(by = 'ID', inplace = True)
sub_stack.reste_index(drop = True, inplace = True)
sub_stack.to_csv('submission_stack.csv', index = False)

sub = sub_stack.copy()
sub['Label'] = sub['Label']*0.5 + submission['Label'] * 0.5
sub.to_csv('submission_ensemble.csv', index = False)