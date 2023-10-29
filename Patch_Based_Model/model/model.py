import torch as t
import torch.nn as nn

class tabularModel(nn.Module):
    def __init__(self, in_dim=5, dropout=0.35):
        """
        Initialize a tabular model.

        Args:
            in_dim (int): Number of input features.
            dropout (float): Dropout rate for regularization.
        """
        super(tabularModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_dim, 64),    # Fully connected layer with input size 'in_dim' and output size 64
            nn.ReLU(),               # ReLU activation function
            nn.Dropout(dropout),     # Dropout layer to prevent overfitting

            nn.Linear(64, 128),      # Another fully connected layer with input size 64 and output size 128
            nn.ReLU(),               # ReLU activation
            nn.Dropout(dropout)      # Dropout
        )

    def forward(self, x):
        """
        Forward pass of the tabular model.

        Args:
            x (tensor): Input data tensor.

        Returns:
            tensor: Output of the model after forward pass.
        """
        return self.feature_extractor(x)

class imageModel(nn.Module):
    def __init__(self, out_dim=128, backbone_name='resnet18', dropout=0.35):
        """
        Initialize an image model.

        Args:
            out_dim (int): Dimensionality of the output feature vector.
            backbone_name (str): Name of the backbone architecture (e.g., 'resnet18').
            dropout (float): Dropout rate for regularization.
        """
        super(imageModel, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(512, 1024),    # Fully connected layer with input size 512 and output size 1024
            nn.ReLU(),               # ReLU activation function
            nn.Dropout(dropout),     # Dropout layer

            nn.Linear(1024, 1024),   # Another fully connected layer with input size 1024 and output size 1024
            nn.ReLU(),               # ReLU activation
            nn.Dropout(dropout),

            nn.Linear(1024, out_dim),  # Fully connected layer with input size 1024 and output size 'out_dim'
            nn.ReLU(),                # ReLU activation
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass of the image model.

        Args:
            x (tensor): Input image data tensor.

        Returns:
            tensor: Output of the model after forward pass.
        """
        x = self.feature_extractor(x)
        return x

class MelanomaPredictor(nn.Module):
    def __init__(self, in_dim=5, out_dim=128, backbone_name='resnet18', dropout=0.35):
        """
        Initialize the Melanoma predictor model.

        Args:
            in_dim (int): Number of input features for tabular data.
            out_dim (int): Dimensionality of the output feature vector for the image model.
            backbone_name (str): Name of the backbone architecture (e.g., 'resnet18') for image feature extraction.
            dropout (float): Dropout rate for regularization.
        """
        super(MelanomaPredictor, self).__init__()

        # Initialize image and tabular feature extractors
        self.image_feature_extractor = imageModel(out_dim=out_dim, backbone_name=backbone_name, dropout=dropout)
        self.tabular_feature_extractor = tabularModel(in_dim=in_dim, dropout=dropout)
        
        # Define the classifier network
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),   # Fully connected layer with input size 256 and output size 512
            nn.ReLU(),             # ReLU activation
            nn.Dropout(dropout),   # Dropout

            nn.Linear(512, 256),   # Another fully connected layer with input size 512 and output size 256
            nn.ReLU(),             # ReLU activation
            nn.Dropout(dropout)
        )

        # Define output layers for breslow thickness, ulceration, and relapse
        self.breslow = nn.Linear(256, 5)  # Output for breslow thickness with 5 classes
        self.ulceration = nn.Linear(256, 1)  # Output for ulceration (binary classification)
        self.relapse = nn.Linear(256, 1)  # Output for relapse (binary classification)
        
        # Sigmoid activation for binary outputs
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, params):
        """
        Forward pass of the Melanoma predictor model.

        Args:
            image (tensor): Input image data tensor.
            params (tensor): Input tabular data tensor.

        Returns:
            tuple: Predictions for relapse, breslow thickness, and ulceration.
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
