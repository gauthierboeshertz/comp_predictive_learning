import os 
import torch 
import numpy as np
from collections import defaultdict
from compo_predictive_learning.metrics.disentanglement import disentanglement_metric
from compo_predictive_learning.metrics.clustering import get_optimal_n_cluster,get_rnn_activities_and_sources_for_loader_for_clustering
from compo_predictive_learning.utils.make_all_plots import make_all_plots
import traceback


def log_metrics(logger,metrics, groups):
    for group_name, keys in groups.items():
        logger(f"--- {group_name} Metrics ---")
        for k in keys:
            if k in metrics:
                logger(f"{k:30s}: {metrics[k][-1]}")
    # any leftovers?
    other = set(metrics) - {k for keys in groups.values() for k in keys}
    if other:
        logger(" --- Other Metrics ---")
        for k in sorted(other):
            logger(f"{k:30s}: {metrics[k]}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_loop(config,
               train_loader,
               val_loader,
               smaller_train_loader,
               analysis_loader,
               model,
               optimizer,
               all_contexts,
               latent_names,
               classification_metric_train_loaders,
               classification_metric_val_loaders,
               logger,
               save_folder = None):
    
    metrics = defaultdict(list)
    training_losses = []
    training_loss = []
    training_accs = []
    regularization_loss = []
    validation_losses = []
    logging_groups =  {}
    
    if config.compute_metrics:
        logging_groups["Classifier Metrics"] = []
        for lat in latent_names:
            logging_groups["Classifier Metrics"] += [f"{lat}_train_classifier_gener",f"{lat}_val_classifier_gener"]
        
        if config.compute_clustering:
            logging_groups["Clustering Metrics"] = ["optimal_n_clusters"]
            
    for step,batch in enumerate(train_loader):
        if step >= config.train_loop.num_steps:
            break
        model.train()
        imgs, latents, contexts = batch
        out =  model.loss(imgs)
        loss,_,rnn_encoding = out[0],out[1],out[2]
        
        if len(out) == 5:
            training_accs.append(out[4].item())
        training_loss.append(loss.item())
        if config.train_loop.pretrain_act_decay > 0:
            loss += config.train_loop.pretrain_act_decay*torch.mean(torch.square(rnn_encoding))
        if config.train_loop.pretrain_act_l1 > 0:
            loss += config.train_loop.pretrain_act_l1*torch.mean(torch.abs(rnn_encoding))
        if config.train_loop.pretrain_weight_l1 > 0:
            for name,param in model.rnn.named_parameters():
                loss += config.train_loop.pretrain_weight_l1*torch.mean(torch.abs(param))
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        regularization_loss.append(loss.item())
        if (step % config.train_loop.eval_every) == 0 or step == config.train_loop.num_steps - 1:
            model.eval()
            val_loss = []
            val_accs = []
            with torch.no_grad():
                for val_batch in val_loader:
                    val_imgs, val_latents, val_contexts = val_batch
                    out = model.loss(val_imgs)
                    loss = out[0]
                    if len(out) == 5:
                        val_accs.append(out[4].item())
                    val_loss.append(loss.item())
                validation_losses.append(np.mean(val_loss))

            logger(f"Step {step+1}/{config.train_loop.num_steps}, "
                        f"Training loss: {np.mean(training_loss):.4f}, "
                        f" {f'Accuracy: {np.mean(training_accs):.4f}, ' if len(training_accs) > 0 else ''}"
                        f"Regularization loss: {np.mean(regularization_loss):.4f}, "
                        f"Validation loss: {np.mean(val_loss):.4f}"
                        f"{f', Validation Accuracy: {np.mean(val_accs):.4f}' if len(val_accs) > 0 else ''}")

            training_losses.append(np.mean(training_loss))
            training_loss = []
            regularization_loss = []
            training_accs = []
            
            if config.compute_metrics and ((step % config.train_loop.compute_metrics_every) == 0 or step == config.train_loop.num_steps - 1):
                for lat in classification_metric_train_loaders.keys():
                    classification_metric_train_loader,lat_idx = classification_metric_train_loaders[lat]
                    classification_metric_val_loader = classification_metric_val_loaders[lat][0]
                    classifier_metrics = disentanglement_metric(config, model, idx_to_classify=lat_idx, train_loader=classification_metric_train_loader, val_loader=classification_metric_val_loader)
                    metrics[f"{lat}_train_classifier_gener"].append(classifier_metrics[0])
                    metrics[f"{lat}_val_classifier_gener"].append(classifier_metrics[1])

                if config.compute_clustering:
                    activity, sources = get_rnn_activities_and_sources_for_loader_for_clustering(model, analysis_loader)
                    try:
                        optimal_n_clusters = get_optimal_n_cluster(model, activations=activity, contexts=sources, time_variance=False, device=DEVICE)[0]
                        metrics["optimal_n_clusters"].append(optimal_n_clusters)

                    except Exception as e:
                        print(f"Error in clustering: {e}")
                        metrics["optimal_n_clusters"].append(0)
                
                log_metrics(logger,metrics, logging_groups)

        if config.make_plots and ((config.train_loop.make_plots_every>0 and step % config.train_loop.make_plots_every == 0) or step == config.train_loop.num_steps - 1):
            try:
                make_all_plots(config,
                            model,
                            all_contexts,
                            train_loader,
                            val_loader,
                            smaller_train_loader,
                            analysis_loader,
                            classification_metric_train_loaders,
                            classification_metric_val_loaders,
                            figure_folders= f"figures_{step}" if save_folder is None else os.path.join(save_folder, f"figures_{step}"))
            except Exception as e:
                logger(f"Error in making plots: {e}")
                logger(traceback.format_exc())

                
        if config.train_loop.save_model_every> 0 and ( (step>0 and  step % config.train_loop.save_model_every == 0) or step == config.train_loop.num_steps - 1):
            save_path = f"{save_folder}/model_step_{step}.pth" if save_folder else f"model_step_{step}.pth"
            torch.save(model.state_dict(),save_path)
            logger(f"Saved model at step {step}.")
    
    logger(f"Training completed after {step} steps.")
    return model,training_losses, validation_losses, metrics
                    