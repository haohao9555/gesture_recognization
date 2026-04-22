import torch
import torch.nn as nn
from torchvision.models import resnet18


class CNNLSTM(nn.Module):
    """
    CNN + LSTM model for video classification.

    Input shape:
    - x: [batch_size, num_frames, 3, 224, 224]

    Main steps:
    1. Merge batch and time dimensions:
       [B, T, C, H, W] -> [B*T, C, H, W]
    2. Extract frame-wise CNN features:
       [B*T, C, H, W] -> [B*T, feature_dim]
    3. Restore sequence structure:
       [B*T, feature_dim] -> [B, T, feature_dim]
    4. Feed features into LSTM:
       [B, T, feature_dim] -> hidden state [num_layers, B, hidden_dim]
    5. Use final hidden state for classification:
       [B, hidden_dim] -> [B, num_classes]
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        backbone = resnet18(weights=None)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.cnn = backbone
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = x.shape

        # [B, T, C, H, W] -> [B*T, C, H, W]
        x = x.view(batch_size * num_frames, channels, height, width)

        # [B*T, C, H, W] -> [B*T, feature_dim]
        frame_features = self.cnn(x)

        # [B*T, feature_dim] -> [B, T, feature_dim]
        frame_features = frame_features.view(batch_size, num_frames, self.feature_dim)

        # LSTM output:
        # output: [B, T, hidden_dim]
        # hidden: [num_layers, B, hidden_dim]
        _, (hidden, _) = self.lstm(frame_features)

        # Take the final layer's hidden state: [B, hidden_dim]
        final_hidden = hidden[-1]
        final_hidden = self.dropout(final_hidden)

        # [B, hidden_dim] -> [B, num_classes]
        logits = self.classifier(final_hidden)
        return logits
