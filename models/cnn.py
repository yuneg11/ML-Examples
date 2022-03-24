from flax import linen as nn


class CNN(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.dropout(x, 0.2)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x