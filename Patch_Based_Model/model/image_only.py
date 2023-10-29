import torchvision
import torch.nn as nn

class ImageOnlyModel(nn.Module):
    def __init__(self, backbone_name='resnet34'):
        """
        Initialize an image-only model for melanoma prediction.

        Args:
            backbone_name (str): Name of the backbone architecture (e.g., 'resnet34' or 'resnet18').
        """
        super(ImageOnlyModel, self).__init__()

        # Initialize the specified backbone architecture with pre-trained weights
        if backbone_name == 'resnet34':
            self.backbone = torchvision.models.resnet34(pretrained=True)
        elif backbone_name == 'resnet18':
            self.backbone = torchvision.models.resnet18(pretrained=True)
        
        # Replace the fully connected layer (classifier) with an identity function
        self.backbone.fc = nn.Identity()

        # Output layers for breslow thickness, ulceration, and relapse
        self.breslow = nn.Linear(512, 5)  # Output for breslow thickness with 5 classes
        self.ulceration = nn.Linear(512, 1)  # Output for ulceration (binary classification)
        self.relapse = nn.Linear(512, 1)  # Output for relapse (binary classification)
        
        # Sigmoid activation for binary outputs
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, params):
        """
        Forward pass of the image-only model.

        Args:
            x (tensor): Input image data tensor.
            params (tensor): Additional parameters (not used in this model).

        Returns:
            tuple: Predictions for relapse, breslow thickness, and ulceration.
        """
        x = self.backbone(x)  # Pass the input through the backbone architecture
        
        breslow = self.breslow(x)  # Predicted breslow thickness
        ulceration = self.sigmoid(self.ulceration(x))  # Predicted ulceration (binary)
        relapse = self.sigmoid(self.relapse(x))  # Predicted relapse (binary)
        
        return (relapse, breslow, ulceration)
