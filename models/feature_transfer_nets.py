import torch
from torch.profiler import profile, record_function, ProfilerActivity

class FeatureProjectionMLP(torch.nn.Module):
    def __init__(self, in_features = None, out_features = None, act_layer = torch.nn.GELU):
        super().__init__()
        
        self.act_fcn = act_layer()

        self.input = torch.nn.Linear(in_features, (in_features + out_features) // 2)
        self.projection = torch.nn.Linear((in_features + out_features) // 2, (in_features + out_features) // 2)
        self.output = torch.nn.Linear((in_features + out_features) // 2, out_features)

    def forward(self, x):
        x = self.input(x)
        x = self.act_fcn(x)

        x = self.projection(x)
        x = self.act_fcn(x)

        x = self.output(x)

        return x
    
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

    model = FeatureProjectionMLP(768,768)
    model.eval()
    inputs = torch.rand(50176, 768)
    with profile(activities=[ProfilerActivity.CPU],
                 profile_memory=True, record_shapes=True) as prof:

        model(inputs)


    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))