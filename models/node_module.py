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

# Example usage
if __name__ == "__main__":
    # Instantiate the model
    model = NodeModule()
    print(model)

    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 768, 32, 32)  # Example input with 768 channels and 32x32 spatial dimensions
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
