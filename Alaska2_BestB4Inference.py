from efficientnet_pytorch import EfficientNet
from glob import glob
from sklearn.model_selection import GroupKFold
from torch import nn
from skimage import io
from datetime import datetime
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import cv2
import torch
import os
import time
import timm
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import sklearn

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

DATA_ROOT_PATH = '/home/chloe/Siting/ALASKA2'
SEED = 42
results = []
submissions = []

# check device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(SEED)

# EfficientNet
def build_model():
    net = timm.create_model('efficientnet_b0', pretrained=True)
    #net = EfficientNet.from_pretrained('efficientnet-b4')
    net._fc = nn.Linear(in_features=1792, out_features=4, bias=True)
    print(net)
    return net

model = build_model().cuda()

#checkpoint = torch.load('checkpoints/best-checkpoint-042epoch_3_c.bin')
#model.load_state_dict(checkpoint['model_state_dict']);
model.eval();

def get_test_transforms(mode):
    if mode == 0:
        return A.Compose([
                A.Resize(height = 512, width = 512, p = 1.0),
                A.ToTensorV2(p = 1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                ], p = 1.0)
    elif mode == 1:
        return A.Compose([
                A.HorizontalFlip(p = 1),
                A.Resize(height = 512, width = 512, p = 1.0),
                A.ToTensorV2(p = 1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                ], p = 1.0)
    elif mode == 2:
        return A.Compose([
                A.VerticalFlip(p = 1),
                A.Resize(height = 512, width = 512, p = 1.0),
                A.ToTensorV2(p = 1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                ], p = 1.0)
    else:
        return A.Compose([
                A.HorizontalFlip(p = 1),
                A.VerticalFlip(p = 1),
                A.Resize(height = 512, width = 512, p = 1.0),
                A.ToTensorV2(p = 1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
                ], p = 1.0)

class DatasetSubmissionRetriever(Dataset):

    def __init__(self, image_names, transforms = None):
        super().__init__()
        self.image_names = image_names
        self.transforms = transforms

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

    def __len__(self) -> int:
        return self.image_names.shape[0]


for mode in range(0, 4):
    dataset = DatasetSubmissionRetriever(
        image_names=np.array([path.split('/')[-1] for path in glob(DATA_ROOT_PATH + '/Test/*.jpg')]),
        transforms=get_test_transforms(mode),
    )

    data_loader = DataLoader(
        dataset,
        batch_size = 8,
        shuffle = False,
        num_workers = 2,
        drop_last = False,
    )

    result = {'Id': [], 'Label': []}
    for step, (image_names, images) in enumerate(data_loader):
        print(step, end='\r')

        y_pred = model(images.cuda())
        y_pred = 1 - nn.functional.softmax(y_pred, dim = 1).data.cpu().numpy()[:, 0]

        result['Id'].extend(image_names)
        result['Label'].extend(y_pred)

    results.append(result)

y_pred = model(images.cuda())
y_pred = 1 - nn.functional.softmax(y_pred, dim=1).data.cpu().numpy()[:, 0]

result['Id'].extend(image_names)
result['Label'].extend(y_pred)

for mode in range(0, 4):
    submission = pd.DataFrame(results[mode])
    submissions.append(submission)

for mode in range(0, 4):
    submissions[mode].to_csv(f'submission_{mode}.csv', index=False)

submissions[0]['Label'] = (submissions[0]['Label'] * 3 + submissions[1]['Label'] * 1 + submissions[2]['Label'] * 1 +
                           submissions[3]['Label'] * 1) / 6
submissions[0].to_csv(f'submission_B4_3_c.csv', index=False)