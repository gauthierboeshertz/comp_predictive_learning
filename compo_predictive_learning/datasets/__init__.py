import random
import torch 
import numpy as np
from .contextual_concat import ContextualConcatenator,ConcatExclusiveSampler,InfiniteConcatExclusiveSampler,InfiniteSampler
from .contextual_sketch import make_sketch_loader,make_contextual_sketch_collate
from .contextual_sketch import make_dataset as make_dataset_contextual_sketch
from .structured_context_generator import create_structured_contexts
from .dsprites import make_DSprites_loader, make_DSprites_collate_fn, load_raw as load_raw_dsprites,make_dataset as make_dataset_contextual_dsprites, TransitionDSprites

def worker_init_fn(worker_id):                                                                                                                                
    seed = 0                
                                                                                                                                   
    torch.manual_seed(seed)                                                                                                                                   
    torch.cuda.manual_seed(seed)                                                                                                                              
    torch.cuda.manual_seed_all(seed)                                                                                          
    np.random.seed(seed)                                                                                                             
    random.seed(seed)                                                                                                       
    torch.manual_seed(seed)                                                                                                                                   
    return

def make_contextual_loader(config,
                           context_vals,
                           num_sample_per_context,
                           context_vector_size,
                           context_start_idx=0,
                           put_in_dict=False,
                           infinite_stream=False,
                           shuffle=True,
                           **kwargs):

    if "sketch" in config.dataset.name:
        datasets,context_vectors = make_sketch_loader(config,context_vals,context_vector_size,context_start_idx,num_sample_per_context,put_in_dict)
        collate_fn = make_contextual_sketch_collate("cuda" if torch.cuda.is_available() else "cpu")
    elif "dsprites" in config.dataset.name:
        datasets,context_vectors = make_DSprites_loader(config,
                                                         context_vals,
                                                         context_vector_size,
                                                         context_start_idx,
                                                         num_sample_per_context,
                                                         put_in_dict,
                                                         imgs = kwargs.get("imgs",None),
                                                         factor_values= kwargs.get("factor_values",None),
                                                         factor_classes= kwargs.get("factor_classes",None))
         
        collate_fn = make_DSprites_collate_fn("cuda" if torch.cuda.is_available() else "cpu")

    else:
        raise ValueError(f"Dataset {config.dataset.name} not supported.")
    
    if put_in_dict:
        dataloaders = {}
        for name,(ds,ctxt_vec) in datasets.items():
            g = torch.Generator()
            g.manual_seed(0)
            dataloaders[name] = torch.utils.data.DataLoader(ContextualConcatenator([ds],ctxt_vec),
                                                            batch_size=config.train_loop.batch_size,
                                                            num_workers=0,
                                                            shuffle=shuffle,
                                                            pin_memory=False,
                                                            worker_init_fn=worker_init_fn,
                                                            generator=g)
        return dataloaders
    else:
        concat_ds = ContextualConcatenator(datasets=datasets,context_for_datasets=context_vectors)
        if config.one_context_per_batch:
            if not infinite_stream:
                sampler = ConcatExclusiveSampler(concat_ds,batch_size=config.train_loop.batch_size,shuffle=shuffle)
            else:
                assert shuffle, "Infinite stream with exclusive sampler requires shuffle=True"
                sampler = InfiniteConcatExclusiveSampler(concat_ds,batch_size=config.train_loop.batch_size)
            g = torch.Generator()
            g.manual_seed(0)

            context_loader = torch.utils.data.DataLoader(concat_ds,
                                                         batch_sampler=sampler,
                                                         num_workers=0,
                                                         pin_memory=False,
                                                         collate_fn=collate_fn,
                                                         worker_init_fn=worker_init_fn,
                                                         generator=g)
        else:
            g = torch.Generator()
            g.manual_seed(0)

            if infinite_stream:
                sampler = InfiniteSampler(concat_ds,shuffle=shuffle)
                context_loader = torch.utils.data.DataLoader(concat_ds,
                                                            batch_size=config.train_loop.batch_size,
                                                            num_workers=0,
                                                            pin_memory=False,
                                                            collate_fn=collate_fn,
                                                            sampler=sampler,
                                                            worker_init_fn=worker_init_fn,
                                                            generator=g)

            else:
                context_loader = torch.utils.data.DataLoader(concat_ds,
                                                            batch_size=config.train_loop.batch_size,
                                                            num_workers=0,
                                                            shuffle=shuffle,
                                                            pin_memory=False,
                                                            collate_fn=collate_fn,
                                                            worker_init_fn=worker_init_fn,
                                                            generator=g)
        return context_loader


def make_sketch_dataloaders(config):

    context_size = 3
    
    train_contexts,val_contexts = create_structured_contexts(config)
    pretrain_loader = make_contextual_loader(config,train_contexts,config.dataset.num_pretrain_drawings_per_context,context_size,0,infinite_stream=True,shuffle=True)
    val_loader = make_contextual_loader(config,val_contexts,config.dataset.num_val_drawings_per_context,context_size,len(train_contexts),shuffle=False)
    smaller_pretrain_loader = make_contextual_loader(config,train_contexts,config.train_loop.batch_size,context_size,0,shuffle=False)

    def make_abstract_loader(train_or_val,test_latent):
        ds = make_dataset_contextual_sketch(1,
                                            config,
                                            task_disentanglement=True,
                                            task_disentanglement_train_set=train_or_val,
                                            task_disentanglement_latent=test_latent)
        
        loader = torch.utils.data.DataLoader(ds,
                                            batch_size=config.train_loop.batch_size,
                                            num_workers=0,
                                            shuffle=False,
                                            drop_last=False,
                                            pin_memory=False)
        return loader
    
    classification_metric_train_loaders = {}
    classification_metric_val_loaders = {}
    latents = []
    for lat_idx, lat in enumerate(["primitive","scale","color","position"]):
        if lat == "scale":
            if len(config.dataset.scales) == 1:
                continue 
        if lat == "color":
            if len(config.dataset.colors) == 1:
                continue
        latents.append(lat)
        classification_metric_train_loaders[lat] = (make_abstract_loader(True,lat),lat_idx) 
        classification_metric_val_loaders[lat] = (make_abstract_loader(False,lat),lat_idx) 
    
    all_contexts = train_contexts + val_contexts 
    analysis_loader = make_contextual_loader(config=config,context_vals=all_contexts,num_sample_per_context=128,context_vector_size=context_size,context_start_idx=0,shuffle=False)

    return pretrain_loader, val_loader, smaller_pretrain_loader,analysis_loader, classification_metric_train_loaders, classification_metric_val_loaders,latents,train_contexts,val_contexts


def make_dsprites_dataloaders(config):
    train_contexts,val_contexts = create_structured_contexts(config)
    ddd_imgs, factor_values, factor_classes = load_raw_dsprites(config.dataset.path)
    context_size = 3
    pretrain_loader = make_contextual_loader(config,train_contexts,config.dataset.num_train_sequences_per_context,context_size,0,infinite_stream=True,shuffle=True,
                                            imgs=ddd_imgs,
                                            factor_values=factor_values,
                                            factor_classes=factor_classes)
    val_loader = make_contextual_loader(config,val_contexts,config.dataset.num_val_sequences_per_context,context_size,len(train_contexts),shuffle=False,
                                                imgs=ddd_imgs,
                                                factor_values=factor_values,
                                                factor_classes=factor_classes)  
    smaller_pretrain_loader = make_contextual_loader(config,train_contexts,config.train_loop.batch_size,context_size,0,shuffle=False,
                                                imgs=ddd_imgs,
                                                factor_values=factor_values,
                                                factor_classes=factor_classes)
    
    def make_abstract_loader(train_or_val,test_latent):
        ds = make_dataset_contextual_dsprites(1,config,
                                            imgs=ddd_imgs,
                                            factor_values=factor_values,
                                            factor_classes=factor_classes,
                                            task_abstract=True,task_abstract_train_set=train_or_val,task_abstract_latent=test_latent)
        loader = torch.utils.data.DataLoader(ds,
                                            batch_size=config.train_loop.batch_size,
                                            num_workers=0,
                                            shuffle=False,
                                            pin_memory=False)
        return loader
    # latents = TransitionDSprites.factors

    classification_metric_train_loaders = {}
    classification_metric_val_loaders = {}
    assert len(config.dataset.task_disentanglement_num_timesteps) == 1, "DSprites disentanglement only supports one timestep for now"
    latents = [lat for lat in TransitionDSprites.factors if not config.dataset["one_"+lat]]
    latents_idx = {}
    for idx, lat in enumerate(TransitionDSprites.factors):
        latents_idx[lat] = idx
    
    print("Latents for dsprites disentanglement:", latents,latents_idx)
    classification_metric_train_loaders = {}
    classification_metric_val_loaders = {}
    for  lat in (latents):
        lat_idx = latents_idx[lat]
        classification_metric_train_loaders[lat] = (make_abstract_loader(True,lat),lat_idx) 
        classification_metric_val_loaders[lat] = (make_abstract_loader(False,lat),lat_idx) 

    print("Not onel aten ts", latents)
    all_contexts = train_contexts + val_contexts
    analysis_loader = make_contextual_loader(config=config,dataset_name="dsprites",context_vals=all_contexts,num_sample_per_context=128,context_vector_size=context_size,context_start_idx=0,shuffle=False)
    
    return pretrain_loader, val_loader, smaller_pretrain_loader,analysis_loader ,classification_metric_train_loaders, classification_metric_val_loaders,latents, train_contexts,val_contexts

