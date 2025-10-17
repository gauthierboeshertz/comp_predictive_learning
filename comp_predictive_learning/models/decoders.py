import torch 
import torch.nn as nn

class ConvDecoder(nn.Module):
    def __init__(self,
                 input_dim, 
                 hidden_dims, 
                 output_dim, 
                 activation="relu",
                 norm_layer="batch_norm",
                 name="ConvDecoder"):

        super(ConvDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.norm_layer = norm_layer
        layers = []
        assert len(hidden_dims)==3 or len(hidden_dims)==4, "ConvDecoder requires 3 or 4 hidden dimensions"
        self.fc_expansion = 8 if len(hidden_dims) == 3 else 4
        
        self.fc = nn.Linear(input_dim, (self.fc_expansion**2)*hidden_dims[0])
        for i in range(1,len(hidden_dims)):
            layers.append(nn.ConvTranspose2d(hidden_dims[i-1], hidden_dims[i], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False))
                
            if norm_layer == "batch_norm":
                layers.append(nn.BatchNorm2d(hidden_dims[i]))
            elif norm_layer == "instance_norm":
                layers.append(nn.InstanceNorm2d(hidden_dims[i]))
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
        self.conv_layers = nn.Sequential(*layers)
        self.output_layer = nn.ConvTranspose2d(hidden_dims[-1], output_dim[0], kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        
    def forward(self, x):
        x = self.fc(x)
        num_timesteps = 0
        if x.dim() > 2:
            num_timesteps = x.size(0)
            x = x.view(x.size(0)*x.size(1), x.size(2))
        x = x.view(x.size(0), -1, self.fc_expansion, self.fc_expansion)
        x = self.conv_layers(x)
        x = self.output_layer(x)
        if num_timesteps > 0:
            x = x.view(num_timesteps, -1, x.size(1), x.size(2), x.size(3))
        return x
    
    def new_instance(self):
        return ConvDecoder(self.input_dim, self.hidden_dims, self.output_dim, self.activation, self.norm_layer)