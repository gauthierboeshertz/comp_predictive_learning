import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
import math
from matplotlib.lines import Line2D
from itertools import combinations
import torch
from compo_predictive_learning.metrics.clustering import get_optimal_n_cluster, get_rnn_activities_and_sources_for_loader_for_clustering, analyze_and_sort_clusters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_context_weights_heatmaps(W, all_neurons_groups, context_to_cluster):

    sets_int = {str(k): np.array(list(map(int, v)), dtype=int)
                for k, v in all_neurons_groups.items()}

    ordered_clusters = []
    for ctx, clusters in context_to_cluster.items():
        valid_clusters = [str(c) for c in clusters if str(c) in sets_int]
        if valid_clusters:
            for cluster_id in valid_clusters:
                ordered_clusters.append((ctx, cluster_id))

    if not ordered_clusters:
        print("No valid clusters found to plot.")
        return {}

    n_clusters = len(ordered_clusters)
    
    def block_weights(src_id, tgt_id):
        src_neurons, tgt_neurons = sets_int[src_id], sets_int[tgt_id]
        if src_neurons.size == 0 or tgt_neurons.size == 0:
            return np.array([])
        return W[np.ix_(src_neurons, tgt_neurons)].ravel()

    
    heatmap_matrix = np.zeros((n_clusters, n_clusters))
    intra_context_weights = []
    inter_context_weights = []
    intra_cluster_weights = []

    for i in range(n_clusters):
        src_ctx, src_id = ordered_clusters[i]
        for j in range(n_clusters):
            tgt_ctx, tgt_id = ordered_clusters[j]

            weights = block_weights(src_id, tgt_id)
            mean_w = np.mean(weights) if weights.size > 0 else 0
            heatmap_matrix[i, j] = mean_w
            if src_id == tgt_id:
                intra_cluster_weights.append(mean_w)
            elif src_ctx == tgt_ctx:
                intra_context_weights.append(mean_w)
            else:
                inter_context_weights.append(mean_w)



    fig, ax = plt.subplots(figsize=(max(8, n_clusters * 0.4), max(6, n_clusters * 0.4)))
    
    v_max = np.max(np.abs(heatmap_matrix))
    im = ax.imshow(heatmap_matrix, cmap='bwr', vmin=-v_max, vmax=v_max, aspect='equal')

    ax.set_xticks(np.arange(n_clusters))
    ax.set_yticks(np.arange(n_clusters))
    ax.set_xticklabels([cid for _, cid in ordered_clusters], rotation=90, fontsize=16)
    ax.set_yticklabels([cid for _, cid in ordered_clusters], fontsize=16)

    boundaries = [i for i in range(n_clusters - 1) if ordered_clusters[i][0] != ordered_clusters[i+1][0]]
    for b in boundaries:
        ax.axhline(b + 0.5,  linewidth=1.5)
        ax.axvline(b + 0.5,  linewidth=1.5)
            
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(label="Mean Weight",fontsize=22)
    fig.tight_layout()

    return {
        "context_weights_heatmaps": (fig, ax),
    }

    
def network_weights_plots(model,loader=None,activations=None,contexts=None,classification_metric_train_loaders=None,classification_metric_val_loaders=None):

    if activations is None or contexts is None:
        assert loader is not None, "Either provide a loader or precomputed activations and contexts"
        activations, contexts = get_rnn_activities_and_sources_for_loader_for_clustering(model, loader)

    max_num_clusters, scores, norm_var_activities_per_context, active_units, labels = get_optimal_n_cluster(model,
                                                                                                        activations=activations,
                                                                                                        contexts=contexts,
                                                                                                        time_variance=False,
                                                                                                        device=DEVICE)
    
    if max_num_clusters <=2:
        print("Not enough clusters found, skipping network weights plots.")
        return {}
    unique_contexts, inv = contexts.unique(dim=0, return_inverse=True)

    (sorted_order, group_labels, group_boundaries,
    peak_map, informative_indices, cluster_profiles) = analyze_and_sort_clusters(
        norm_var_activities_per_context=norm_var_activities_per_context.cpu().numpy(),
        contexts_unique=unique_contexts.cpu(),
        labels=labels,
        selectivity_threshold= 0.2,
        purity_threshold= 0.5
    )
    
    original_indices = np.where(active_units)[0]
    cluster_to_neurons = {}
    for i, cluster_id in enumerate(np.unique(labels)):
        neurons_in_cluster_mask = labels == cluster_id
        cluster_to_neurons[cluster_id] = original_indices[neurons_in_cluster_mask]

    all_neurons_groups = {}
    for k in cluster_to_neurons:
        all_neurons_groups[str(k)] = cluster_to_neurons[k]

    for k in all_neurons_groups:
        print(f"Group {k} has {(all_neurons_groups[k])} neurons")
        
    context_to_cluster = {}
    curr_idx = 0
    for i, gl in enumerate(group_labels):
        context_to_cluster[gl] = []
        for k in range(curr_idx, group_boundaries[i]):
            context_to_cluster[gl].append(sorted_order[k])
        curr_idx = group_boundaries[i]

    recurrent_weight = model.rnn.state_dict()["weight_hh.weight"].cpu().numpy()
    all_fig_axs = {}    
    
    context_weights_fig_axs = plot_context_weights_heatmaps(recurrent_weight, all_neurons_groups, context_to_cluster)
    
    all_fig_axs.update(context_weights_fig_axs)
    return all_fig_axs
    