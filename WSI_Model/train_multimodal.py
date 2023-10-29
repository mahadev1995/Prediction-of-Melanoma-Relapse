# Import necessary libraries and modules
import os
import random
import torch as t
import pandas as pd
from utils.loss import FocalLoss
from utils.trainer import Trainer 
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.dataset import MultimodalDataset
from model.model import MelanomaPredictor
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set random seeds for reproducibility
random.seed(0)
t.manual_seed(42)

# Define paths and hyper-parameters
path = './data/new/'
batch_size = 64      
learning_rate = 1e-4
start_epoch = 0
end_epoch = 50
gamma = 4
pos_wt = 1
lamda = 2
modelName = 'resnet34'
ckpt_path = f'./checkpoints/{modelName}_multimodal_new2/'
history_path = f'./history/{modelName}_multimodal_new2/'
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

# Create the neural network model (MelanomaPredictor)
net = MelanomaPredictor(backbone_name=modelName, dropout=0.2).cuda()
net = t.nn.DataParallel(net)

# Create training and validation datasets using MultimodalDataset class
train_dataset = MultimodalDataset(path=path + 'train_data.csv')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

val_dataset = MultimodalDataset(path=path + 'val_data.csv')
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

# Define loss functions, optimizer, and learning rate scheduler
criterion1 = FocalLoss(gamma=gamma, pos_weight=pos_wt)
optimizer = t.optim.Adam(net.parameters(), lr=learning_rate)
criterion2 = t.nn.CrossEntropyLoss()
criterion3 = t.nn.BCELoss()

scheduler = ReduceLROnPlateau(optimizer, patience=7, verbose=True, factor=0.2)

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
