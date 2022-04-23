from torch import nn


class CNN(nn.Sequential):
    def __init__(self, num_classes, image_shape=(3, 32, 32)):
        c, h, w = image_shape
        feature_dim = (h // 4) * (w // 4) * 64

        super().__init__(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=feature_dim, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes),
        )
