import torch
import torch.nn as nn

class NodeModule(nn.Module):
    def __init__(self):
        super(NodeModule, self).__init__()
        # Single Conv2d layer: input channels=768, output channels=64, kernel size=1
        self.conv = nn.Conv2d(in_channels=768, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv_2 = nn.Conv2d(in_channels=768, out_channels=64, kernel_size=1, stride=1, padding=0)

    def forward(self, x, y):
        return self.conv(x), self.conv_2(y)


class ChannelAttentionReduction(nn.Module):
    def __init__(self, in_channels=192, out_channels=64, reduction_ratio=4):  # Corrected __init__
        super(ChannelAttentionReduction, self).__init__()  # Corrected initialization
        reduced_channels = in_channels // reduction_ratio  # Reduce for attention

        # Squeeze-and-Excitation (SE) Style Attention
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # (B, C, 1, 1)
        self.fc1 = nn.Linear(in_channels, reduced_channels)  # Reduce channels
        self.fc2 = nn.Linear(reduced_channels, in_channels)  # Restore channels
        self.sigmoid = nn.Sigmoid()

        # 1x1 Convolution to Reduce Channels (192 → 64)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # Channel Attention (SE Block)
        y = self.global_avg_pool(x).view(B, C)  # (B, C)
        y = torch.nn.functional.relu(self.fc1(y))  # Reduce channels
        y = self.fc2(y)  # Restore channels
        y = self.sigmoid(y).view(B, C, 1, 1)  # Reshape for broadcasting

        # Apply channel attention
        x = x * y  # (B, C, H, W)

        # Reduce channels using 1x1 convolution
        x = self.conv1x1(x)  # (B, 64, H, W)

        return x


# Example usage
if __name__ == "__main__":
    # Instantiate the model
    model = NodeModule()
    print(model)

    # # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # dummy_input = torch.randn(1, 768, 32, 32)  # Example input with 768 channels and 32x32 spatial dimensions
    # output = model(dummy_input)

    # print(f"Input shape: {dummy_input.shape}")
    # print(f"Output shape: {output.shape}")
    # model = ChannelAttention()
    # print(model)
    # dummy_input = torch.randn(1, 192, 224, 224)  # Example input with 192 channels and 32x32 spatial dimensions
