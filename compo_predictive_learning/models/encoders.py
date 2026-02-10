import torch 
import torch.nn as nn

class ConvEncoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dims, 
                 output_dim, 
                 activation="relu",
                 norm_layer="batch_norm",
                 max_pooling=False,
                 name="ConvEncoder"):
        super(ConvEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.norm_layer = norm_layer
        layers = []
        conv_stride = 2 if not max_pooling else 1
        for i, hidden_dim in enumerate(hidden_dims):

            if i == 0:
                layers.append(nn.Conv2d(input_dim[0], hidden_dim, kernel_size=3, stride=conv_stride, padding=1, bias=False))
            else:
                layers.append(nn.Conv2d(hidden_dims[i-1], hidden_dim, kernel_size=3, stride=conv_stride, padding=1, bias=False))
            
            if norm_layer == "batch_norm":
                layers.append(nn.BatchNorm2d(hidden_dim))
            elif norm_layer == "instance_norm":
                layers.append(nn.InstanceNorm2d(hidden_dim))
            elif norm_layer == "none":
                pass
            else:
                raise ValueError("Norm function not supported")        
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                raise ValueError("Activation function not supported")   
                 
            if max_pooling:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_layers = nn.Sequential(*layers)
        fc_input_size = self.conv_layers(torch.zeros(1, *input_dim)).view(1, -1).size(1)
        print("Output shape from conv layers", fc_input_size)
        self.fc = nn.Linear(fc_input_size, output_dim)
        self.output_dim = output_dim
        
    def forward(self, x):
        has_time_idx = x.dim() > 4
        num_timesteps = 0
        if has_time_idx:
            assert x.dim() == 5, "Input tensor must be 5D (T, B, C, H, W) if it has a time dimension"
            num_timesteps = x.size(0)
            x = x.reshape(x.size(0)*x.size(1), x.size(2), x.size(3), x.size(4))

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        if has_time_idx:
            x = x.reshape(num_timesteps, -1, x.size(1))
        return x
    
    def new_instance(self):
        return ConvEncoder(self.input_dim, self.hidden_dims, self.output_dim, self.activation, self.norm_layer)
    
    
    def get_output_size(self, input_dim):
        return self.fc(self.conv_layers(torch.zeros(1, *input_dim)).view(1, -1)).size(1)