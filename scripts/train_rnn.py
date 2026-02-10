import os 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch 
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
from omegaconf import OmegaConf
import torch 
import numpy as np
import random
from compo_predictive_learning.models import create_model
import hydra
import logging
from collections import defaultdict
from compo_predictive_learning.datasets import make_dsprites_dataloaders, make_sketch_dataloaders
from compo_predictive_learning.utils.train_loop import train_loop
from compo_predictive_learning.utils.make_all_plots import make_all_plots
logger = logging.getLogger(__name__)

def set_seed(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

np.set_printoptions(precision=5, suppress=True, linewidth=200)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def get_activity_and_context(config,model,loader,subsample_activites=0):

    activities = []
    contexts = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            prim_sequence_inputs,latents,context = batch
            inputs = prim_sequence_inputs
            _,context_activity,activity,_ = model(inputs,context)
            if subsample_activites > 0:
                assert config.one_context_per_batch, "Subsampling activities only works with one context per batch"
                num_videos_sampled = int(activity.shape[1]*subsample_activites)
                activity = activity[:,:num_videos_sampled,:]
                context = context[:num_videos_sampled,:]
            activities.append(activity.cpu())
            contexts.append(context.cpu())
    
    activities = torch.cat(activities,dim=1).permute(1,0,2)
    contexts = torch.cat(contexts,dim=0)
    print(f"Activities shape: {activities.shape}, Contexts shape: {contexts.shape}")
    return activities, contexts


@hydra.main(config_path="configs",config_name="train_rnn")
def main(config):

    set_seed(config.seed)   
    if "sketch" in config.dataset.name:
        pretrain_loader,val_loader, smaller_pretrain_loader,analysis_loader,classification_metric_train_loaders, classification_metric_val_loaders,latent_names, train_contexts,val_contexts = make_sketch_dataloaders(config)
    
    elif "dsprites" in config.dataset.name:
        pretrain_loader, val_loader, smaller_pretrain_loader,analysis_loader ,classification_metric_train_loaders, classification_metric_val_loaders,latent_names, train_contexts,val_contexts = make_dsprites_dataloaders(config)
    else:
        raise NotImplementedError(f"Dataset {config.dataset.name} not implemented")
    
    all_contexts = train_contexts + val_contexts

    model = create_model(config).to(DEVICE)

    logger.info(f"Device: {DEVICE}")
    if config.no_redo:
        if os.path.exists("results.npz"):
            files = os.listdir(".")
            model_files = [f for f in files if f.startswith("model") and f.endswith(".pth")]
            sorted_model_files = sorted(model_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            model.load_state_dict(torch.load(sorted_model_files[-1],map_location=DEVICE))
            logger.info("=========  Loaded existing model =========")
            make_all_plots(config,model,all_contexts,pretrain_loader,val_loader,smaller_pretrain_loader,analysis_loader,classification_metric_train_loaders, classification_metric_val_loaders,"figures_redo")
                
            logger.info("Results already exist, skipping")
            quit()
     
    print(OmegaConf.to_yaml(config))        
    
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.train_loop.pretrain_lr,weight_decay=config.train_loop.pretrain_decay)

    logger.info("-----------------")
    logger.info("Pretraining")
    logger.info("-----------------")
    
    metrics = defaultdict(list)
    
    logging_groups =  {}
    if config.compute_metrics:
        logging_groups["Classifier Metrics"] = []
        for lat in latent_names:
            logging_groups["Classifier Metrics"] += [f"{lat}_train_classifier_gener",f"{lat}_val_classifier_gener"]
        
        if config.compute_clustering:
            logging_groups["Clustering Metrics"] = ["optimal_n_clusters_time_var", "optimal_n_clusters"]
        
    model,training_losses, validation_losses, metrics = train_loop(config,
                                                                pretrain_loader,
                                                                val_loader,
                                                                smaller_pretrain_loader,
                                                                analysis_loader,
                                                                model,
                                                                optimizer,
                                                                all_contexts,
                                                                latent_names,
                                                                classification_metric_train_loaders,
                                                                classification_metric_val_loaders,
                                                                logger=logger.info)
    np.savez("results.npz",
             training_losses=training_losses,
             validation_losses=validation_losses,
             **metrics)
    



if __name__ == "__main__":

    main()