import os
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import itertools as it
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms import functional
from torchvision.transforms import InterpolationMode
from torchvision.models.vision_transformer import vit_b_16
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from modules import modules
from modules import losses

# utility function
to_tuple = lambda x: tuple(2*[x])
def convert_to_img(tensor):
    C, W, H = tensor.shape
    # tensor = functional.normalize(tensor,
    #                               mean= C*[-1],
    #                               std= C*[2],
    #                               inplace=True)
    return tensor.permute(1,2,0)



# Define dataset class
class tvtLaneDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform = None, target_transform = None, validation = False):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.target_transform = target_transform
        self.len = self.count_files(target_dir)
        self.validation = validation
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        target_files = self.all_files_in(self.target_dir)
        target_path = next(it.islice(target_files, idx, idx+1), None)
        target = Image.open(target_path)
        if self.target_transform:
            target = self.target_transform(target)
            kernel_size = 3
            with torch.no_grad():
                target = nn.functional.max_pool2d(target,
                                                kernel_size = kernel_size,
                                                stride = 1,
                                                padding = (kernel_size-1)//2)
        input_path = None
        if not self.validation:
            splited_path = target_path.split('/')
            fold_name = splited_path[-1].split('.')[0]
            street_name = splited_path[-2]
            image_number = splited_path[-3].split('_')[-2]
            class_image = splited_path[-4]
            input_path = os.path.join(self.input_dir,class_image, street_name, fold_name,image_number+".jpg")
        else:
            splited_path = target_path.split('/')
            arq_name = splited_path[-1]
            fold_name = splited_path[-2]

            folds_diferent_treat = ['0530', "0531", "0601"]

            if fold_name not in folds_diferent_treat:
                input_path = os.path.join(self.input_dir, fold_name, arq_name)
            else:
                image_number = splited_path[-3].split('_')[-2]
                input_path = os.path.join(self.input_dir, fold_name, arq_name.split('.')[0], image_number+".jpg")

        input = Image.open(input_path)

        if self.transform:
            input = self.transform(input)

        return input, target

    def all_files_in(self,path,ignore_list = []):
        for entry in os.scandir(path):
            for ignore in ignore_list:
                if ignore in entry.path:
                    continue
            if entry.is_dir(follow_symlinks=False):
                yield from self.all_files_in(entry.path)
            elif entry.is_file():
                yield entry.path

    def count_files(self,root_path):
        qtd = 0
        dirs = os.listdir(root_path)
        for dir in dirs:
            current_dir = os.path.join(root_path, dir)
            if os.path.isfile(current_dir):
                qtd += 1
            elif os.path.isdir(current_dir):
                qtd += self.count_files(current_dir)
        return qtd




    

# create Dataloader
# O path deve ser modificado para onde está o trainset e o testset

interpolation_mode = InterpolationMode.NEAREST
img_size = 224

train_dataset = tvtLaneDataset(
                input_dir='../Datasets/trainset/image',
                target_dir='../Datasets/trainset/truth',
                transform= v2.Compose([
                    v2.ToImage(),
                    v2.Resize(to_tuple(img_size),
                              interpolation=interpolation_mode),
                    v2.ToDtype(torch.float32, scale = True),
                ]),
                target_transform= v2.Compose([
                    v2.ToImage(),
                    v2.Resize(to_tuple(img_size),
                              interpolation=interpolation_mode),
                    v2.ToDtype(torch.float32, scale = True),
                ])
            )

test_dataset = tvtLaneDataset(
                validation= True,
                input_dir='../Datasets/testset/image',
                target_dir='../Datasets/testset/truth',
                transform= v2.Compose([
                    v2.ToImage(),
                    v2.Resize(to_tuple(img_size),
                              interpolation=interpolation_mode),
                    v2.ToDtype(torch.float32, scale = True),
                ]),
                target_transform= v2.Compose([
                    v2.ToImage(),
                    v2.Resize(to_tuple(img_size),
                              interpolation=interpolation_mode),
                    v2.ToDtype(torch.float32, scale = True),
                ])
            )


## Modelo

class Model(nn.Module):
    def __init__(self, vit):
        super(Model, self).__init__()
        self.encoder = modules.ViT_Encoder(vit)
        self.decoder = modules.Decoder(vit.hidden_dim)
    def forward(self,x):
        x_encoded = self.encoder(x)
        return self.decoder(x_encoded,x)
    
model = Model(vit_b_16())

#hiperparams

learning_rate = 1e-3
batch_size = 32
epochs = 150
weight_decay = 1e-4
momentum = 0.9

train_dataloader = DataLoader(
    train_dataset,
    batch_size= batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=2
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size= batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=2
)

#optimizer

optimizer = torch.optim.SGD(
    model.parameters(),
    lr = learning_rate,
    weight_decay= weight_decay,
    momentum= momentum,
    )


#Loss
class Loss(nn.Module):
    def __init__(self):
        super(Loss,self).__init__()
        self.dice_loss = losses.DiceLoss()
        self.focal_loss = losses.FocalLoss()
    def forward(self,y_hat,y):
        return self.dice_loss(y_hat,y)/3 + self.focal_loss(y_hat,y)


loss_fn = Loss()

#######TRAIN#############

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Used Device: {device}")

def train_loop(dataloader, model, loss_fn, optimizer, device, scaler):
    model = model.to(device)
    model.train()
    local_loss = 0
    global_loss = 0

    for batch, (X,y) in enumerate(dataloader):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)

        #compute prediction and loss
        with torch.autocast(device_type=str(device), dtype=torch.float16):
          pred = model(X)
          loss = loss_fn(pred,y)

        #backpropagation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        local_loss += loss.item()
        global_loss += loss.item()

        if (batch+1) % 200 == 0:
            local_loss = local_loss/200
            print(f"loss:{local_loss:e}")
            local_loss = 0

    global_loss = global_loss/len(dataloader)
    print(f"TRAIN GLOBAL LOSS: {global_loss:e}\n")
    return global_loss


############# TEST ###################

def test_loop(dataloader, model, loss_fn,device):
    model = model.to(device)
    model.eval()
    acc_loss = 0
    with torch.no_grad():
        for batch, (X,y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred,y)
            acc_loss += loss.item()

    acc_loss /= len(dataloader)
    print(f"TEST GLOBAL LOSS: {acc_loss:e}\n")
    return acc_loss


############ TRAIN/TEST LOOP ####################
scaler = torch.amp.GradScaler()

ratio = 1e-1
patience = 7
current_patience = patience
base = np.inf
model_dict = None
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn,optimizer, device, scaler)
    test_loss = test_loop(test_dataloader, model, loss_fn, device)

    if current_patience == 0:
      break;

    if test_loss > base*(1-ratio):
        current_patience -= 1
    else:
        current_paciente = patience
        base = test_loss
        torch.save(model.state_dict(), "model.pth")
        print("SAVED PYTORCH STATE DICT IN model.pth")

print("Done!")
