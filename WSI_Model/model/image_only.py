import torchvision
import torch.nn as nn

class ImageOnlyModel(nn.Module):
    def __init__(self, backbone_name='resnet34'):
        """
        A custom image-only neural network model for a specific task.

        Args:
            backbone_name (str): Backbone architecture name, 'resnet34' or 'resnet18'.
        """
        super(ImageOnlyModel, self).__init__()

        # Initialize the backbone architecture based on the specified name
        if backbone_name == 'resnet34':
            self.backbone = torchvision.models.resnet34(pretrained=True)
        elif backbone_name == 'resnet18':
            self.backbone = torchvision.models.resnet18(pretrained=True)
        else:
            raise ValueError("Unsupported backbone_name. Choose 'resnet34' or 'resnet18'.")

        # Replace the fully connected layer (fc) with an identity layer
        self.backbone.fc = nn.Identity()

        # Define additional layers for the specific task
        self.breslow = nn.Linear(512, 5)  # Output for breslow thickness prediction
        self.ulceration = nn.Linear(512, 1)  # Output for ulceration prediction
        self.relapse = nn.Linear(512, 1)  # Output for relapse prediction
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary predictions

    def forward(self, x, params):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor.
            params: Additional parameters (not used in this model).

        Returns:
            Tuple[torch.Tensor]: Predictions for relapse, breslow thickness, and ulceration.
        """
        x = self.backbone(x)
        
        # Generate predictions
        breslow = self.breslow(x)
        ulceration = self.sigmoid(self.ulceration(x))
        relapse = self.sigmoid(self.relapse(x))
        
        return (relapse, breslow, ulceration)
