# Import necessary libraries and modules
import os
import torch as t
import numpy as np
import pandas as pd
from model.image_only import ImageOnlyModel
import torchvision.transforms as transforms

# Define the path to the checkpoint file
ckpt_path =  '/home/woody/iwso/iwso089h/VisioMel/seminar/fullimage/checkpoints/resnet34_imageonly_bce_loss_16_new/checkpoint_12.ckp'

# Initialize the ImageOnlyModel and load the checkpoint
net = ImageOnlyModel(backbone_name='resnet34').cuda()
net = t.nn.DataParallel(net)
ckp = t.load(ckpt_path)
net.load_state_dict(ckp['state_dict'])
net.eval()

# Define a transformation to preprocess images
transform = transforms.Compose(
    [transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]
                         )]
)

# Load train, validation, and test datasets
train_df = pd.read_csv('../data/data_splits/train/train_labels.csv')
val_df = pd.read_csv('../data/data_splits/val/val_labels.csv')
test_df = pd.read_csv('../data/data_splits/test/test_labels.csv')

# Print the first few rows of the train dataset
print(train_df.head())

# Extract features from training images
image_feature_extractor = net.module.backbone
image_feature_extractor.fc = t.nn.Identity()

filenames = train_df.filename.values
train_image_features = []

print('Extracting Features for training images.')

for filename in filenames:
    features = []
    features.append(filename)

    relapse = train_df[train_df['filename'] == filename]['relapse'].values[0]

    img_name = os.path.splitext(filename)[0]+'.npy'
    img_path = f'/home/vault/iwso/iwso089h/images/{relapse.item()}/{img_name}'
    image = np.load(img_path)
    image = transform(image)[None].cuda()

    image_features = image_feature_extractor(image)[0].detach().cpu().numpy()
    features.extend(list(image_features))
    train_image_features.append(features)

# Convert and save training image features as numpy array
train_image_features_np = np.array(train_image_features)
np.save('./data/new/train_image_features_new2.npy', train_image_features_np)

# Extract features from validation images
val_images_features = []
filenames = val_df.filename.values

print('Extracting features for validation images.')

for filename in filenames:
    features = []
    features.append(filename)

    relapse = val_df[val_df['filename'] == filename]['relapse'].values[0]

    img_name = os.path.splitext(filename)[0]+'.npy'
    img_path = f'/home/vault/iwso/iwso089h/images/{relapse.item()}/{img_name}'
    image = np.load(img_path)
    image = transform(image)[None].cuda()

    image_features = image_feature_extractor(image)[0].detach().cpu().numpy()
    features.extend(list(image_features))
    val_images_features.append(features)

# Convert and save validation image features as numpy array
val_images_features_np = np.array(val_images_features)
np.save('./data/new/val_images_features_new2.npy', val_images_features_np)

# Extract features from test images
test_images_features = []
filenames = test_df.filename.values

print('Extracting features for testing images.')

for filename in filenames:
    features = []
    features.append(filename)

    relapse = test_df[test_df['filename'] == filename]['relapse'].values[0]

    img_name = os.path.splitext(filename)[0]+'.npy'
    img_path = f'/home/vault/iwso/iwso089h/images/{relapse.item()}/{img_name}'
    image = np.load(img_path)
    image = transform(image)[None].cuda()

    image_features = image_feature_extractor(image)[0].detach().cpu().numpy()
    features.extend(list(image_features))
    test_images_features.append(features)

# Convert and save test image features as numpy array
test_images_features_np = np.array(test_images_features)
np.save('./data/new/test_images_features_new2.npy', test_images_features)
