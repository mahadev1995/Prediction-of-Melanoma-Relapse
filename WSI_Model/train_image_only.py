# Import necessary libraries and modules
import os
import random
import torch as t
import pandas as pd
from utils.trainer import Trainer 
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.dataset import MelanomaDataset
from model.image_only import ImageOnlyModel
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set random seeds for reproducibility
random.seed(0)
t.manual_seed(42)

# Define paths and hyper parameters
data_path = '/home/janus/iwso-datasets/visiomel'
batch_size = 16      
learning_rate = 1e-4
start_epoch = 0
end_epoch = 50
lamda = 1
modelName = 'resnet34'
size = 1024
ckpt_path = f'./checkpoints/{modelName}_imageonly_bce_loss_16_new/'
history_path = f'./history/{modelName}_imageonly_bce_loss_16_new/'
save_interval = 1

# Create checkpoint and history directories if they do not exist
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
    print('Created Directory: ', ckpt_path)
else:
    print('Directory is already there!!')

if not os.path.exists(history_path):
    os.makedirs(history_path)
    print('Created Directory: ', history_path)
else:
    print('Directory is already there!!')

# Load metadata
meta_data = pd.read_csv('../data/meta_data/meta_data_new.csv')

# Extract specific columns from metadata
columns = ['filename', 'breslow', 'ulceration', 'relapse']
meta_labels = meta_data[columns]

# Remove columns from metadata
meta_data = meta_data.drop(columns=columns[1:])

# Preprocess numeric columns in metadata
meta_data["age"] = meta_data["age"].div(100).round(2)
meta_data["body_site"] = meta_data["body_site"].div(14).round(2)

# Print a snippet of the metadata
print(meta_data.head())

# Load training and validation labels from CSV files
train_labels = pd.read_csv('../data/data_splits/train/train_labels.csv')
val_labels = pd.read_csv('../data/data_splits/val/val_labels.csv')

# Create the neural network model (ImageOnlyModel)
net = ImageOnlyModel(backbone_name=modelName).cuda()
net = t.nn.DataParallel(net)

# Define data augmentation transformations for training and validation datasets
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90, interpolation=transforms.InterpolationMode.BILINEAR, expand=True),
    transforms.RandomCrop((size, size), pad_if_needed=True),
    transforms.ColorJitter(),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create training and validation datasets using MelanomaDataset class
train_dataset = MelanomaDataset(train_labels, meta_data, meta_labels, data_path, transforms=transform_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

val_dataset = MelanomaDataset(val_labels, meta_data, meta_labels, data_path, transforms=transform_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# Define loss functions, optimizer, and learning rate scheduler
criterion1 = t.nn.BCELoss()
criterion2 = t.nn.CrossEntropyLoss()
criterion3 = t.nn.BCELoss()

optimizer = t.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=3e-6)
scheduler = ReduceLROnPlateau(optimizer, patience=10, verbose=True, factor=0.1)

# Create a Trainer instance for training the model
trainer = Trainer(net, criterion1, criterion2, criterion3, scheduler,
                  optimizer, train_loader, val_loader, True, False)

# Start training the model
loss_history, avg_val_loss_relapse = trainer.train(start_epoch=start_epoch, 
                                                   end_epoch=end_epoch,
                                                   checkpoint_interval=save_interval, 
                                                   checkpoint_path=ckpt_path,
                                                   history_path=history_path,
                                                   lamda=lamda
                                                  )

# Create a DataFrame to store the loss history and save it to a CSV file
loss_df = pd.DataFrame(loss_history,
                       columns=['TrainLoss', 'LossRelapse', 'LossMeta', 'ValLoss', 'ValLossRelapse', 'ValLossMeta', 'ValAcc', 'ValF1', 'ValAucRuc'])
loss_df.to_csv(history_path + '_' + str(start_epoch) + '.csv', index=False)
