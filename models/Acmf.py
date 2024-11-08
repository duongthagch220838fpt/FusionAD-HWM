import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.layers import ConvNormAct,create_act_layer,get_act_layer,make_divisible

class ACMF(nn.Module):
    def __init__(self, in_channels_img, in_channels_evt):
        super(ACMF, self).__init__()
        self.conv1 = nn.Conv2d(in_channels_img + in_channels_evt, in_channels_img + in_channels_evt, kernel_size=1)
        self.ca = ChannelAttention(in_channels_evt)
        self.sa = SpatialAttention()
        self.conv2 = nn.Conv2d(in_channels_img + in_channels_evt, in_channels_img + in_channels_evt, kernel_size=1)

    def forward(self, F_img_t, F_evt_t):
        # Merge image and event features
        x = torch.cat([F_img_t, F_evt_t], dim=0)
        print(x.shape)
        x = self.conv1(x)

        # Apply channel-attention and spatial-attention
        evt_att = self.ca(F_evt_t)
        evt_att = self.sa(evt_att)

        # Enhance the fused features
        x = x * evt_att
        x = self.conv2(x)

        # Sum the enhanced features
        out = F_img_t + x
        return out

# Using timm implement of CBAM
class ChannelAttention(nn.Module):
    """ Original CBAM channel attention module, currently avg + max pool variant only.
    """
    def __init__(
            self, channels, rd_ratio=1./16, rd_channels=None, rd_divisor=1,
            act_layer=nn.ReLU, gate_layer='sigmoid', mlp_bias=False):
        super(ChannelAttention, self).__init__()
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = nn.Conv2d(channels, rd_channels, 1, bias=mlp_bias)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, 1, bias=mlp_bias)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_avg = self.fc2(self.act(self.fc1(x.mean((2, 3), keepdim=True))))
        x_max = self.fc2(self.act(self.fc1(x.amax((2, 3), keepdim=True))))
        return x * self.gate(x_avg + x_max)


class LightChannelAttn(ChannelAttention):
    """An experimental 'lightweight' that sums avg + max pool first
    """
    def __init__(
            self, channels, rd_ratio=1./16, rd_channels=None, rd_divisor=1,
            act_layer=nn.ReLU, gate_layer='sigmoid', mlp_bias=False):
        super(LightChannelAttn, self).__init__(
            channels, rd_ratio, rd_channels, rd_divisor, act_layer, gate_layer, mlp_bias)

    def forward(self, x):
        x_pool = 0.5 * x.mean((2, 3), keepdim=True) + 0.5 * x.amax((2, 3), keepdim=True)
        x_attn = self.fc2(self.act(self.fc1(x_pool)))
        return x * F.sigmoid(x_attn)


class SpatialAttention(nn.Module):
    """ Original CBAM spatial attention module
    """
    def __init__(self, kernel_size=7, gate_layer='sigmoid'):
        super(SpatialAttention, self).__init__()
        self.conv = ConvNormAct(2, 1, kernel_size, apply_act=False)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_attn = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        x_attn = self.conv(x_attn)
        return x * self.gate(x_attn)
    

class LightSpatialAttn(nn.Module):
    """An experimental 'lightweight' variant that sums avg_pool and max_pool results.
    """
    def __init__(self, kernel_size=7, gate_layer='sigmoid'):
        super(LightSpatialAttn, self).__init__()
        self.conv = ConvNormAct(1, 1, kernel_size, apply_act=False)
        self.gate = create_act_layer(gate_layer)

<<<<<<< HEAD
    def forward(self, x):
        x_attn = 0.5 * x.mean(dim=1, keepdim=True) + 0.5 * x.amax(dim=1, keepdim=True)
        x_attn = self.conv(x_attn)
        return x * self.gate(x_attn)
    
if __name__ == '__main__':
=======
class ACMF(nn.Module):
    def __init__(self, in_channels_img, in_channels_evt, reduction=16):
        super(ACMF, self).__init__()
        # Initial convolution to merge image and event features
        self.conv1 = nn.Conv2d(
            in_channels_img + in_channels_evt,
            in_channels_img,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        # Average pooling to downscale features
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Channel and spatial attention modules
        self.ca = ChannelAttention(in_channels_img, reduction=reduction)
        self.sa = SpatialAttention()

        # Final convolution to refine enhanced features
        self.conv2 = nn.Conv2d(
            in_channels_img,
            in_channels_img,
            kernel_size=3,
        )

        # self.conv3 = nn.Conv2d(in_channels_evt)

    def forward(self, F_img_t, F_evt_t):
        # Concatenate image and event features along the channel dimension
        cat_fea = torch.cat(
            (F_img_t, F_evt_t), dim=1
        )  # Shape: (batch, in_channels_img + in_channels_evt, H, W)

        # Apply the first convolution
        cat_fea = self.conv1(cat_fea)
        # print(f"after conv1 {cat_fea.shape}")
        # Apply average pooling
        cat_fea = self.avg_pool(cat_fea)  # Shape: bx768x1x1
        # print(cat_fea.shape)
        # Either split tensor or just multiply Element-wise
        F_evt_t_mul = F_evt_t * cat_fea
        F_img_t_mul = F_img_t * cat_fea
        # Apply channel attention and spatial attention sequentially
        ca_out = self.ca(F_evt_t_mul)  # Channel attention output
        sa_out = self.sa(ca_out)  # Spatial attention output

        # Multiply the attention maps with the input feature map
        x = F_img_t_mul + sa_out

        # Apply the second convolution
        x = self.conv2(x)
        evt = self.conv2(F_img_t_mul)
        # Upsample to match the original spatial dimensions of `F_img_t`
        # x = F.interpolate(
        #     x, size=F_img_t.shape[2:], mode="bilinear", align_corners=False
        # )

        # Element-wise sum with the original image features
        out = evt + x
        out_upsample = torch.nn.functional.interpolate(out, size=(224, 224), mode='bilinear', align_corners=False)
        out = out_upsample.permute(0, 2, 3, 1) # (batch, H, W, 768)
        return out


if __name__ == "__main__":
>>>>>>> 645e482 ()
    # Define a dummy input for testing
    batch_size = 1
    in_channels_img = 768
    in_channels_evt = 768
    height = 256
    width = 256

    F_img_t = torch.randn(batch_size, 768)
    F_evt_t = torch.randn(batch_size, 768)

    # Initialize the ACMF module
    acmf = ACMF(in_channels_img, in_channels_evt)
    # Test the forward pass
    output = acmf(F_img_t, F_evt_t)
    print(output.shape) # [50176,768]
