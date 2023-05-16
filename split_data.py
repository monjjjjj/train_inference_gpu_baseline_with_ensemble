from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms
from torchvision import datasets
import glob
import os
import shutil
from sklearn.model_selection import train_test_split

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_mean_and_std(train_data):
    train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=1, shuffle=False, num_workers=6,
            pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X,_ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())

if __name__ == '__main__':
    train_dataset = ImageFolder(root = r'/home/chloe/Siting/ALASKA2', transform = transforms.ToTensor())
    print(get_mean_and_std(train_dataset))

image_list = glob.glob('/home/chloe/Siting/ALASKA2/*/*.jpg')
#print(image_list)

file_dir = 'ALASKA2_data_sp91'
if os.path.exists(file_dir):
    print('true')
    shutil.rmtree(file_dir)
    os.makedirs(file_dir)
else:
    os.makedirs(file_dir)


trainval_files, val_files = train_test_split(image_list, test_size = 0.1, random_state = 42)
train_dir = 'train'
val_dir = 'val'
train_root = os.path.join(file_dir, train_dir)
val_root = os.path.join(file_dir, val_dir)
for file in trainval_files:
    file_class = file.replace("\\","/").split('/')[-2]
    file_name = file.replace("\\","/").split('/')[-1]
    file_class = os.path.join(train_root, file_class)
    if not os.path.isdir(file_class):
        os.makedirs(file_class)
    shutil.copy(file, file_class + '/' + file_name)

for file in val_files:
    file_class = file.replace("\\","/").split('/')[-2]
    file_name = file.replace("\\","/").split('/')[-1]
    file_class = os.path.join(val_root, file_class)
    if not os.path.isdir(file_class):
        os.makedirs(file_class)
    shutil.copy(file, file_class + '/' + file_name)