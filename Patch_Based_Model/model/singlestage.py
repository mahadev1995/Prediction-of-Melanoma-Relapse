import torchvision
import torch as t
import torch.nn as nn

class tabularModel(nn.Module):
    def __init__(self, in_dim=5, dropout=0.35):
        """
        Initialize a tabular model for melanoma prediction.

        Args:
            in_dim (int): Input feature dimension.
            dropout (float): Dropout probability.
        """
        super(tabularModel, self).__init__()
        self.feature_extractor = nn.Sequential(
                            nn.Linear(in_dim, 64),
                            nn.ReLU(),
                            nn.Dropout(dropout),

                            nn.Linear(64, 128),
                            nn.ReLU(),
                            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass of the tabular model.

        Args:
            x (tensor): Input tabular data tensor.

        Returns:
            tensor: Extracted features from the tabular data.
        """
        return self.feature_extractor(x)

class imageModel(nn.Module):
    def __init__(self, out_dim=128, backbone_name='resnet18', dropout=0.35):
        """
        Initialize an image model for melanoma prediction.

        Args:
            out_dim (int): Output feature dimension.
            backbone_name (str): Name of the backbone architecture (e.g., 'resnet34' or 'resnet18').
            dropout (float): Dropout probability.
        """
        super(imageModel, self).__init__()

        # Initialize the specified backbone architecture with pre-trained weights
        if backbone_name == 'resnet34':
            self.backbone = torchvision.models.resnet34(pretrained=True)
        elif backbone_name == 'resnet18':
            self.backbone = torchvision.models.resnet18(pretrained=True)

        # Replace the fully connected layer (classifier) with an identity function
        self.backbone.fc = nn.Identity()

        self.feature_extractor = nn.Sequential(
                                nn.Linear(512, 1024),
                                nn.ReLU(),
                                nn.Dropout(dropout),

                                nn.Linear(1024, 1024),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                
                                nn.Linear(1024, out_dim),
                                nn.ReLU(),
                                nn.Dropout(dropout))
        
    def forward(self, x):
        """
        Forward pass of the image model.

        Args:
            x (tensor): Input image data tensor.

        Returns:
            tensor: Extracted features from the image data.
        """
        x = self.backbone(x)
        x = self.feature_extractor(x)
        return x


class MelanomaPredictor(nn.Module):
    def __init__(self, in_dim=5, out_dim=128, backbone_name='resnet18', dropout=0.35):
        """
        Initialize a melanoma predictor model combining tabular and image features.

        Args:
            in_dim (int): Input feature dimension for the tabular data.
            out_dim (int): Output feature dimension for the image data.
            backbone_name (str): Name of the backbone architecture (e.g., 'resnet34' or 'resnet18').
            dropout (float): Dropout probability.
        """
        super(MelanomaPredictor, self).__init__()

        self.image_feature_extractor = imageModel(out_dim=out_dim, backbone_name=backbone_name, dropout=dropout)
        self.tabular_feature_extractor = tabularModel(in_dim=in_dim, dropout=dropout)
        self.classifier = nn.Sequential(
                                nn.Linear(256, 512),
                                nn.ReLU(),
                                nn.Dropout(dropout), 

                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Dropout(dropout)) 
        
        self.breslow = nn.Linear(256, 5)  # Output for breslow thickness with 5 classes
        self.ulceration = nn.Linear(256, 1)  # Output for ulceration (binary classification)
        self.relapse = nn.Linear(256, 1)  # Output for relapse (binary classification)
        self.sigmoid = nn.Sigmoid()


    def forward(self, image, params):
        """
        Forward pass of the melanoma predictor model.

        Args:
            image (tensor): Input image data tensor.
            params (tensor): Input tabular data tensor.

        Returns:
            tuple: Predictions for relapse, breslow thickness, and ulceration.
        """
        image_embedidng = self.image_feature_extractor(image)
        tabular_embedding = self.tabular_feature_extractor(params)

        embedding = t.cat([image_embedidng.view(-1, 128), tabular_embedding.view(-1, 128)], 1)
        embedding = self.classifier(embedding)

        breslow = self.breslow(embedding)
        ulceration = self.ulceration(embedding)
        relapse = self.relapse(embedding)

        ulceration = self.sigmoid(ulceration)
        relapse = self.sigmoid(relapse)

        return (relapse, breslow, ulceration)
