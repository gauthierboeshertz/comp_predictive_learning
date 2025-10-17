import torch
import torch.nn as nn 
from typing import List, Tuple,Optional
from torch import Tensor
import hydra

class RNNAE(nn.Module):
    def __init__(self, 
                type: str,
                encoder_cfg,
                decoder_cfg,
                rnn_cfg,
                name="rnn_ae"):
        super(RNNAE, self).__init__()
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.rnn_cfg = rnn_cfg
        self.encoder = hydra.utils.instantiate(encoder_cfg)
        self.rnn = hydra.utils.instantiate(rnn_cfg)
        self.latent_dim = self.rnn.hidden_dim
        self.name = name
        self.decoder = hydra.utils.instantiate(decoder_cfg)
        self.type = type
        self.loss_fn = self.new_loss_fn()
        self.activation = nn.Identity()         
        
    def forward(self, input:Tensor):
        # type: (Tensor,Tensor) -> Tuple[List[Tensor],Tensor]
        cnn_encoded = self.encoder(input)
        rnn_out = self.rnn(cnn_encoded)
        return self.decoder(rnn_out),cnn_encoded,rnn_out,None
    
    def loss(self, data):
        outputs,cnn_encoded,rnn_out,enc_context = self(data)
        if self.type == "pred":
            context_offset = 1
            labels = data[context_offset+1:]
            outputs = outputs[context_offset:-1]
        else:
            labels = data
        loss = self.loss_fn(outputs,labels)
                
        return loss, cnn_encoded,rnn_out,enc_context

    def create_lesioned_instance(self,indices_to_lesion):
        new_network = self.create_new_instance()
        new_network.rnn.weight_hh.weight.data[:,indices_to_lesion] = 0
        new_network.decoder.fc.weight.data[:,indices_to_lesion] = 0
        return new_network
    
    def new_loss_fn(self,reduction="mean"):
        return nn.MSELoss(reduction=reduction)
        
    def create_new_instance(self,new_params = {},copy_weights=True):
        params = {
                'encoder_cfg':self.encoder_cfg,
                'decoder_cfg': self.decoder_cfg,
                'rnn_cfg': self.rnn_cfg,
                'type': self.type,
                "name": self.name}
        params.update(new_params)
        new_model = RNNAE(**params)
        if copy_weights:
            new_model.load_state_dict(self.state_dict())
        return new_model
