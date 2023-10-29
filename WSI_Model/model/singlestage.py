import torchvision
import torch as t
import torch.nn as nn

class tabularModel(nn.Module):
    def __init__(self, in_dim=5, dropout=0.35):
        """
        Tabular data feature extraction model.

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
            x (torch.Tensor): Input tabular data.

        Returns:
            torch.Tensor: Extracted features.
        """
        return self.feature_extractor(x)

class imageModel(nn.Module):
    def __init__(self, out_dim=128, backbone_name='resnet18', dropout=0.35):
        """
        Image data feature extraction model.

        Args:
            out_dim (int): Output feature dimension.
            backbone_name (str): Backbone architecture name, e.g., 'resnet18'.
            dropout (float): Dropout probability.
        """
        super(imageModel, self).__init__()

        if backbone_name == 'resnet34':
            self.backbone = torchvision.models.resnet34(pretrained=True)
        elif backbone_name == 'resnet18':
            self.backbone = torchvision.models.resnet18(pretrained=True)
        else:
            raise ValueError("Backbone not supported.")

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
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass of the image model.

        Args:
            x (torch.Tensor): Input image data.

        Returns:
            torch.Tensor: Extracted features.
        """
        x = self.backbone(x)
        x = self.feature_extractor(x)
        return x

class MelanomaPredictor(nn.Module):
    def __init__(self, in_dim=5, out_dim=128, backbone_name='resnet18', dropout=0.35):
        """
        Combined model for melanoma prediction using both tabular and image data.

        Args:
            in_dim (int): Input dimension for tabular data.
            out_dim (int): Output dimension for image data feature extraction.
            backbone_name (str): Backbone architecture name for image feature extraction.
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
            nn.Dropout(dropout)
        )

        self.breslow = nn.Linear(256, 5)
        self.ulceration = nn.Linear(256, 1)
        self.relapse = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, params):
        """
        Forward pass of the combined model.

        Args:
            image (torch.Tensor): Input image data.
            params (torch.Tensor): Input tabular data.

        Returns:
            Tuple[torch.Tensor]: Predictions for relapse, breslow thickness, and ulceration.
        """
        image_embedding = self.image_feature_extractor(image)
        tabular_embedding = self.tabular_feature_extractor(params)

        embedding = t.cat([image_embedding.view(-1, 128), tabular_embedding.view(-1, 128)], 1)
        embedding = self.classifier(embedding)

        breslow = self.breslow(embedding)
        ulceration = self.ulceration(embedding)
        relapse = self.relapse(embedding)

        ulceration = self.sigmoid(ulceration)
        relapse = self.sigmoid(relapse)

        return (relapse, breslow, ulceration)
