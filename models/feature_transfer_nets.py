import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

class FeatureProjectionMLP(nn.Module):
    # def __init__(self, in_features = None, out_features = None, act_layer = torch.nn.GELU):
    #     super().__init__()
        
    #     self.act_fcn = act_layer()
    #     self.pool = torch.nn.AdaptiveAvgPool2d((1,1))
    #     self.input = torch.nn.Linear(192, 576)
    #     self.projection = torch.nn.Linear(576, 192)
    #     self.output = torch.nn.Linear(192, 64)
    #     self.drop_out = torch.nn.Dropout(p=0.3)

    # def forward(self, x):

    #     # x = self.pool(x)
    #     print(x.shape)

    #     # x = x.squeeze(-1).squeeze(-1)  # 
    #     print(x.shape)
    #     x = self.input(x)

    #     x = self.act_fcn(x)
    #     x = self.drop_out(x)
    #     x = self.projection(x)
    #     x = self.act_fcn(x)
    #     x = self.drop_out(x)

    #     x = self.output(x)
    #     print(x.shape)
    #     return x
    def __init__(self, in_channels = 192, out_channels = 64, kernel_size=7):
        super(FeatureProjectionMLP, self).__init__()
        self.channel_reduction = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Step 1: Channel Reduction
        x_reduced = self.channel_reduction(x)  # Shape: (batch_size, out_channels, H, W)

        # Step 2: Compute Spatial Attention Map
        max_pool = torch.max(x_reduced, dim=1, keepdim=True).values  # Max pooling along channels
        avg_pool = torch.mean(x_reduced, dim=1, keepdim=True)       # Average pooling along channels
        pooled = torch.cat([max_pool, avg_pool], dim=1)             # Concatenate along channel axis

        # Step 3: Apply Convolution and Sigmoid
        spatial_attention_map = self.spatial_attention(pooled)  # Shape: (batch_size, 1, H, W)

        # Step 4: Apply Attention
        output = x_reduced * spatial_attention_map  # Element-wise multiplication
        print(output.shape)
        return output
    
class FeatureProjectionMLP_big(torch.nn.Module):
    def __init__(self, in_features = None, out_features = None, act_layer = torch.nn.GELU):
        super().__init__()
        
        self.act_fcn = act_layer()

        self.input = torch.nn.Linear(in_features, (in_features + out_features) // 2)
        
        self.projection_a = torch.nn.Linear((in_features + out_features) // 2, (in_features + out_features) // 2)
        self.projection_b = torch.nn.Linear((in_features + out_features) // 2, (in_features + out_features) // 2)
        self.projection_c = torch.nn.Linear((in_features + out_features) // 2, (in_features + out_features) // 2)
        self.projection_d = torch.nn.Linear((in_features + out_features) // 2, (in_features + out_features) // 2)
        self.projection_e = torch.nn.Linear((in_features + out_features) // 2, (in_features + out_features) // 2)

        self.output = torch.nn.Linear((in_features + out_features) // 2, out_features)

    def forward(self, x):
        x = self.input(x)
        x = self.act_fcn(x)

        x = self.projection_a(x)
        x = self.act_fcn(x)
        x = self.projection_b(x)
        x = self.act_fcn(x)
        x = self.projection_c(x)
        x = self.act_fcn(x)
        x = self.projection_d(x)
        x = self.act_fcn(x)
        x = self.projection_e(x)
        x = self.act_fcn(x)

        x = self.output(x)

        return x
    

if __name__ == '__main__':

    model = FeatureProjectionMLP(192,64, 7)
    model.eval()
    inputs = torch.rand(1, 192, 224, 224)
    with profile(activities=[ProfilerActivity.CPU],
                 profile_memory=True, record_shapes=True) as prof:

        model(inputs)


    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))