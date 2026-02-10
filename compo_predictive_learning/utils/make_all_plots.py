import os 
import torch 

import matplotlib.pyplot as plt
import traceback

from datetime import datetime
from compo_predictive_learning.metrics.clustering import plot_cluster_and_lesions,plot_sorted_cluster
from compo_predictive_learning.metrics.network_weights_plots import network_weights_plots
from compo_predictive_learning.metrics.network_activity_plots import network_activity_plots

def savefig(folder,name,fig):
    if not os.path.exists(folder):
        os.makedirs(folder)
    foldi = os.path.join(folder,name)
    if not os.path.exists(foldi):
        os.makedirs(foldi)
    
    print(os.getcwd())
    print(os.path.join(foldi,f"{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}.png"))
    fig.savefig(os.path.join(foldi,f"{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}.png"), bbox_inches='tight', dpi=300,format='png')
    fig.savefig(os.path.join(foldi,f"{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}.pdf"), bbox_inches='tight', dpi=300,format='pdf')
    plt.close(fig)

COLORS = ['#264653', '#2a9d8f', '#ffb703']
COLOR_PREDICTIVE =  COLORS[1]
COLORS_AUTO = COLORS[2]
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


def make_recons_plots(model,loader,fig,ax):
    
    batch = next(iter(loader))
    prim_sequence_inputs, latents,_ = batch
    recons = model(prim_sequence_inputs)[0].detach()

    for b in range(3):
        for t in range(3):
            original_row = b * 2
            recon_row = b * 2 + 1

            ax[original_row, t].imshow(prim_sequence_inputs[t+1, b].permute(1, 2, 0).cpu().numpy())
            
            if model.loss_fn_name == "mse" or model.loss_fn_name == "l1":
                out = recons[t, b].permute(1, 2, 0).cpu().clamp(0, 1).numpy()
            else:
                out = recons[t, b].permute(1, 2, 0).cpu().sigmoid().numpy()
            ax[recon_row, t].imshow(out)
            
            if t == 0:
                ax[original_row, t].set_ylabel(f"Original", fontsize=12, labelpad=10)
                ax[recon_row, t].set_ylabel(f"Reconstruction", fontsize=12, labelpad=10)

            if original_row == 0:
                ax[original_row, t].set_title(f"t={t}", fontsize=14)
                
            ax[original_row, t].set_xticks([])
            ax[original_row, t].set_yticks([])
            ax[recon_row, t].set_xticks([])
            ax[recon_row, t].set_yticks([])

    return fig,ax


def make_all_plots(config,
                   model,
                   all_contexts,
                   pretrain_loader,
                   val_loader,
                   smaller_pretrain_loader,
                   analysis_loader,
                   classification_metric_train_loaders,
                   classification_metric_val_loaders,
                   figure_folders="figures"):

    
    got_clusters = True
    try :
        fig_cluster,ax_cluster,fig_lesion,ax_lesion,les_figs,les_axs,context_for_lesion= plot_cluster_and_lesions(config,model,analysis_loader)
        savefig(figure_folders,"neuron_clusters",fig_cluster)
        savefig(figure_folders,"neuron_clusters_lesion",fig_lesion)
        for i,fig in enumerate(les_figs):
            savefig(figure_folders,f"neuron_clusters_lesion_{i}_{context_for_lesion}",fig)

        
    except Exception as e:
        got_clusters = False
        print(f"Error in clustering: {e}")
        print(traceback.format_exc())
    if not got_clusters:
        print("Skipping clustering plots due to previous error.")
        return
    
    weight_figs_axs_dict = network_weights_plots(model,
                                     loader=analysis_loader,
                                     classification_metric_train_loaders=classification_metric_train_loaders,
                                     classification_metric_val_loaders=classification_metric_val_loaders)
    
    for name, (fig, ax) in weight_figs_axs_dict.items():
        savefig(figure_folders, name, fig)

    
    act_figs_axs_dict = network_activity_plots(model,
                                     loader=analysis_loader,
                                     noiseless=True)
    for name, (fig, ax) in act_figs_axs_dict.items():
        savefig(figure_folders, name, fig)
    
    return 

        