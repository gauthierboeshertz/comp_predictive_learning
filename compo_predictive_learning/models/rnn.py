import torch 
import torch.nn as nn 
import torch.jit as jit
from typing import List
from torch import Tensor
import math

import functools
activation_dict = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "leaky_relu": nn.LeakyReLU,
    "identity": nn.Identity,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "softmax": functools.partial(nn.Softmax, dim=-1),
}

class JitLeakyRNNLayer(jit.ScriptModule):
    def __init__(self, 
                 input_dim, 
                 hidden_dim,
                 activation=nn.Sigmoid,
                 leak_alpha=1,
                 noise=0,
                 rnn_init='orthogonal',
                 mlp_dynamics=False,
                 name=None):
        super(JitLeakyRNNLayer, self).__init__()
        self.input_dim = input_dim
        self.leak_alpha = leak_alpha
        self.noise = noise
        self.rnn_init = rnn_init
        self.hidden_dim = hidden_dim
        
        self.mlp_dynamics = mlp_dynamics
        self.input_layer = nn.Linear(input_dim,hidden_dim)

        if self.mlp_dynamics:
            layer1 = nn.Linear(hidden_dim,hidden_dim)
            layer2 = nn.Linear(hidden_dim,hidden_dim,bias=False)
            if rnn_init == 'orthogonal':
                torch.nn.init.orthogonal_(layer1.weight)
                torch.nn.init.orthogonal_(layer2.weight)
            elif rnn_init == 'eye':
                torch.nn.init.eye_(layer1.weight)/2
                torch.nn.init.eye_(layer2.weight)/2
            else:
                raise ValueError('Invalid rnn_init')
            self.weight_hh = nn.Sequential(layer1,activation(),layer2)
        else:
            self.weight_hh = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
            if rnn_init == 'orthogonal':
                torch.nn.init.orthogonal_(self.weight_hh.weight)
            elif rnn_init == 'eye':
                torch.nn.init.eye_(self.weight_hh.weight)
            elif rnn_init == "unif":
                std = 1/math.sqrt(hidden_dim)
                self.weight_hh.weight.data.uniform_(-std, std)
            elif rnn_init == "zeros":
                self.weight_hh.weight.data.fill_(0)
            else:
                raise ValueError('Invalid rnn_init')
            
        self.alpha = leak_alpha
        self.noise = noise
        self.rnn_act = activation
        self.activation = activation_dict[activation]()

    @jit.script_method
    def forward(self, input:Tensor):
        # type: (Tensor) -> Tensor
        # input is seq_len x batch_size x input_dim
        state = torch.zeros(input.size(1), self.hidden_dim).to(input.device)
        outputs = torch.jit.annotate(List[Tensor], [])
        inputs_dot = self.input_layer(input) 
        if self.noise > 0:
            inputs_dot = inputs_dot + torch.randn_like(inputs_dot)*self.noise
        out = torch.zeros_like(state).to(input.device)
        for i in range((inputs_dot).shape[0]):
            state = state + self.alpha * (-state + inputs_dot[i] + self.weight_hh(out))
            out = self.activation(state)
            outputs += [out]
        return torch.stack(outputs)

    def create_new_instance(self,new_params = {},copy_weights=True):
        params = {'input_dim':self.input_dim,
                'hidden_dim':self.hidden_dim,
                'activation':self.rnn_act,
                'leak_alpha':self.leak_alpha,
                'noise':self.noise,
                'rnn_init':self.rnn_init,
                'mlp_dynamics':self.mlp_dynamics,
                'use_norm_layer':self.use_norm_layer}
        params.update(new_params)
        new_model = JitLeakyRNNLayer(**params)
        if copy_weights:
            new_model.load_state_dict(self.state_dict())
        return new_model
