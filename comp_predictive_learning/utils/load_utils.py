import torch

def unpack_batch(batch,is_conv,device,concat_context=False):
    
    data = batch[0].to(device)
    latents = batch[1].to(device)
    if len(batch)>2:
        context = batch[2].to(device)
        if context.dim() == 2:
            context = context.unsqueeze(1).expand(-1,data.shape[1],-1)
    else:
        context = None
    
    data = data.transpose(0,1)
    if context is not None:
        context = context.transpose(0,1)

    if len(latents.shape) >2:
        latents = latents.permute(1,0,2)
    return data, latents,context
