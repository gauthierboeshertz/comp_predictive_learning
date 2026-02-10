from numpy import prod
import hydra

def create_model(config):
    
    if "sketch" in config.dataset.name :
        num_input_channels = 3 if (len(config.dataset.colors)>1 or config.dataset.colorize_white) else 1
        width = config.dataset.canvas_size
        height = config.dataset.canvas_size
        config.encoder.input_dim = [num_input_channels,width,height]
        config.decoder.output_dim = [num_input_channels,width,height]
        config.rnn.input_dim = config.encoder.output_dim
        config.decoder.input_dim = config.rnn.hidden_dim
        
    elif "dsprites" in config.dataset.name :
        num_input_channels = 1
        width = 64
        height = 64
        config.encoder.input_dim = [num_input_channels,width,height]
        config.decoder.output_dim = [num_input_channels,width,height]
        config.rnn.input_dim = config.encoder.output_dim
        config.decoder.input_dim = config.rnn.hidden_dim

    else:
        raise NotImplementedError(f"Dataset {config.dataset.name} not implemented")
    
    if "cnnae" in config.model._target_.lower():
        config.encoder.input_dim = [num_input_channels,width,height]
        
        config.decoder.input_dim = config.encoder.output_dim
        config.decoder.output_dim = [num_input_channels,width,height]
        return hydra.utils.instantiate(config.model, 
                                        encoder_cfg=config.encoder, 
                                        decoder_cfg=config.decoder,
                                        _recursive_=False)

        
    elif "rnnae" in config.model._target_.lower():
        config.encoder.input_dim = [num_input_channels,width,height]
        config.rnn.input_dim = config.encoder.output_dim
        
        config.decoder.input_dim = config.rnn.hidden_dim
        config.decoder.output_dim = [num_input_channels,width,height]
        return hydra.utils.instantiate(config.model, 
                                        encoder_cfg=config.encoder, 
                                        decoder_cfg=config.decoder,
                                        rnn_cfg=config.rnn,
                                        _recursive_=False)


    model = hydra.utils.instantiate(config.model, encoder_cfg=config.encoder, decoder_cfg=config.decoder,_recursive_=False)
    return model