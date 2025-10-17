import torch.nn as nn 
import torch
from typing import List, Tuple,Optional
from torch import Tensor
import hydra

act_dict = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "identity": nn.Identity,
}

class CNNAE(nn.Module):
    def __init__(self, 
                encoder_cfg,
                decoder_cfg,
                activation="relu",
                noise=0.1,
                type="cnn_ae",
                name="cnn_ae"):
        super(CNNAE, self).__init__()
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.latent_dim = self.encoder_cfg.output_dim
        self.act = activation
        self.encoder = hydra.utils.instantiate(encoder_cfg)
        self.decoder = hydra.utils.instantiate(decoder_cfg)
        self.loss_fn = nn.MSELoss()
        self.activation = nn.Identity() 
        self.noise = noise
        self.activation = act_dict.get(activation, nn.ReLU)()
            
    def forward(self, input:Tensor,context:Optional[Tensor]=None):
        # type: (Tensor,Tensor) -> Tuple[List[Tensor],Tensor]
        encoded = self.encoder(input)
        encoded = self.activation(encoded + torch.randn_like(encoded) * self.noise) # Adding noise to the encoded representation
        return self.decoder(encoded),encoded,encoded,None
        
    def loss(self, data,context,latents):
        outputs,_,encoded,_ = self(data,context)
        loss = 0
        loss += self.loss_fn(outputs,data)
        return loss, encoded, encoded, None

    def create_new_instance(self,new_params = {},copy_weights=True):
        params = {
                'encoder_cfg':self.encoder_cfg,
                'decoder_cfg': self.decoder_cfg,
                'type': self.type,
                'activation': self.act}
        params.update(new_params)
        new_model = CNNAE(**params)
        if copy_weights:
            new_model.load_state_dict(self.state_dict())
        return new_model