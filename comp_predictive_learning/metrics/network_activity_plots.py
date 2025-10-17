import torch
import numpy as np
import matplotlib.pyplot as plt
from comp_predictive_learning.metrics.clustering import get_optimal_n_cluster, get_rnn_activities_and_sources_for_loader_for_clustering, analyze_and_sort_clusters
import copy 
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy

@torch.no_grad()
def plot_mean_cluster_activity(model,
                            loader,
                            cluster_to_neurons,
                            cluster_order,
                            context_to_cluster,
                            noiseless=True,
                            num_samples=4,
                            DEVICE='cuda'):
    
    unique_context_groups = sorted(context_to_cluster.keys())
    base_colors = sns.color_palette("husl", n_colors=len(unique_context_groups))
    context_group_to_base_color = {group_id: color for group_id, color in zip(unique_context_groups, base_colors)}
    cluster_to_group_map = {cid: gid for gid, cids in context_to_cluster.items() for cid in cids}

    valid_cluster_order = [cid for cid in cluster_order if cid in cluster_to_neurons]
    fig, axs = plt.subplots(1, num_samples, figsize=(8 * num_samples, 7), sharey=True)
    axs = np.atleast_1d(axs).ravel()

    if noiseless:
        rnn_cfg = copy.deepcopy(model.rnn_cfg)
        rnn_cfg.noise = 0.0
        plot_model = model.create_new_instance(new_params={'rnn_cfg': rnn_cfg}).to(DEVICE)
    else:
        plot_model = model

    context_showed = []
    for b, (inputs, _, contexts) in enumerate(loader):
        if b >= num_samples:
            break
        
        inputs = inputs.to(DEVICE)
        _, _, activities, _ = plot_model(inputs)
        
        batch_mean_activity = activities.mean(dim=1).cpu().numpy()

        cluster_mean_traces = {}
        cluster_activeness = {}
        for cluster_id in valid_cluster_order:
            neuron_indices = cluster_to_neurons.get(cluster_id, [])
            if len(neuron_indices) > 0:
                mean_trace = batch_mean_activity[:, neuron_indices].mean(axis=1)
                cluster_mean_traces[cluster_id] = mean_trace
                cluster_activeness[cluster_id] = mean_trace.mean()

        most_active_cluster_per_group = {}
        for group_id, cluster_ids_in_group in context_to_cluster.items():
            valid_clusters_in_group = [cid for cid in cluster_ids_in_group if cid in cluster_activeness]
            if not valid_clusters_in_group:
                continue
            
            most_active_id = max(valid_clusters_in_group, key=lambda cid: cluster_activeness[cid])
            most_active_cluster_per_group[group_id] = most_active_id
        
        for cluster_id in valid_cluster_order:
            if cluster_id in cluster_mean_traces:
                mean_trace = cluster_mean_traces[cluster_id]
                group_id = cluster_to_group_map.get(cluster_id)
                
                color = 'gray' 
                if group_id in context_group_to_base_color:
                    base_color = context_group_to_base_color[group_id]
                    is_most_active = most_active_cluster_per_group.get(group_id) == cluster_id
                    
                    if is_most_active:
                        color = base_color 
                    else:
                        color = sns.light_palette(base_color, n_colors=5)[1]

                axs[b].plot(mean_trace, label=f"Cluster {cluster_id}", linewidth=2.5, color=color)

        axs[0].set_xlabel("Time Step",fontsize=24)
        axs[0].set_ylabel("Mean Activity", fontsize=24)
        axs[b].grid(True, linestyle='--', alpha=0.6)
        axs[b].legend(loc='upper right', fontsize='small') 
        context_showed.append([int(c) for c in contexts[0,:3]])

    fig.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.show()
    return fig, axs, context_showed

def _collect_cluster_mean_vectors(model,
                                  loader,
                                  cluster_to_neurons,
                                  DEVICE='cuda',
                                  noiseless=True,
                                  max_batches=None):
    valid_clusters = [cid for cid, idxs in cluster_to_neurons.items() if len(idxs) > 0]
    cluster_vectors = {cid: [] for cid in valid_clusters}

    if noiseless:
        rnn_cfg = copy.deepcopy(model.rnn_cfg)
        rnn_cfg.noise = 0.0
        noiseless_model = model.create_new_instance(new_params={'rnn_cfg': rnn_cfg}).to(DEVICE)
    else:   
        noiseless_model = model
    with torch.no_grad():
        for b_idx, (inputs, _, contexts) in enumerate(loader):
            inputs = inputs.to(DEVICE)
            contexts = contexts.to(DEVICE) 
            _, _, activities, _ = noiseless_model(inputs)  # (T, B, N)

            acts = activities.detach().cpu().numpy()  # (T, B, N)
            for cid in valid_clusters:
                neuron_idx = cluster_to_neurons[cid]
                mean_tb = acts[1:, :, neuron_idx].mean(axis=2)     # (T, B)
                cluster_vectors[cid].append(mean_tb.reshape(-1))  # (T*B,)

            if max_batches is not None and (b_idx + 1) >= max_batches:
                break

    return {cid: np.concatenate(vs, axis=0) for cid, vs in cluster_vectors.items() if len(vs) > 0}


def plot_global_cluster_correlation_with_context_blocks(model,
                                                        loader,
                                                        cluster_to_neurons,
                                                        context_to_cluster,
                                                        DEVICE='cuda',
                                                        noiseless=True,
                                                        max_batches=None,
                                                        show_cluster_ids=False,
                                                        draw_cell_grid=True,
                                                        cell_linewidth=0.4,
                                                        cell_alpha=0.2,
                                                        boundary_linewidth=2.0):
    """
    Builds an all-cluster correlation heatmap, ordered by context. Draws:
      • thin grid lines between EVERY cluster (cell grid)
      • thicker lines at context boundaries
    """
    cluster_vectors = _collect_cluster_mean_vectors(
        model, loader, cluster_to_neurons, noiseless=noiseless,DEVICE=DEVICE, max_batches=max_batches
    )
    if len(cluster_vectors) < 2:
        raise ValueError("Need at least two non-empty clusters to compute correlations.")

    seen = set()
    groups = []
    for ctx, clist in context_to_cluster.items():
        grp = [c for c in clist if c in cluster_vectors and c not in seen]
        if grp:
            groups.append((ctx, grp))
            seen.update(grp)
    leftovers = [c for c in cluster_vectors.keys() if c not in seen]
    if leftovers:
        groups.append(("Unassigned", sorted(leftovers)))

    ordered_clusters = [c for _, grp in groups for c in grp]
    if len(ordered_clusters) < 2:
        raise ValueError("Fewer than two clusters after grouping; cannot plot.")

    min_len = min(len(cluster_vectors[c]) for c in ordered_clusters)
    X = np.stack([cluster_vectors[c][:min_len] for c in ordered_clusters], axis=0)
    C = np.corrcoef(X)

    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    im = ax.imshow(C, interpolation='nearest', aspect='equal',cmap='bwr', vmin=-1, vmax=1)
    
    n = len(ordered_clusters)

    if draw_cell_grid:
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which='minor', linestyle='-', linewidth=cell_linewidth, alpha=cell_alpha)
        ax.tick_params(which='minor', bottom=False, left=False)

    counts = [len(grp) for _, grp in groups]
    edges = np.cumsum([0] + counts) 
    for b in edges:
        ax.axhline(b - 0.5, linewidth=boundary_linewidth)
        ax.axvline(b - 0.5, linewidth=boundary_linewidth)


    n_clusters = len(ordered_clusters)
    ax.set_xticks(np.arange(n_clusters))
    ax.set_yticks(np.arange(n_clusters))
    ax.set_xticklabels([cid for cid in ordered_clusters], rotation=90, fontsize=21)
    ax.set_yticklabels([cid for cid in ordered_clusters], fontsize=21)


    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", fontsize=28)

    fig.tight_layout()
    return fig,ax


def network_activity_plots(model,loader=None,activations=None,contexts=None,noiseless=True):

    if activations is None or contexts is None:
        assert loader is not None, "Either provide a loader or precomputed activations and contexts"
        activations, contexts = get_rnn_activities_and_sources_for_loader_for_clustering(model, loader)

    max_num_clusters, scores, norm_var_activities_per_context, active_units, labels = get_optimal_n_cluster(model,
                                                                                                        activations=activations,
                                                                                                        contexts=contexts,
                                                                                                        time_variance=False)
    unique_contexts, inv = contexts.unique(dim=0, return_inverse=True)
    if max_num_clusters <=2:
        print("Not enough clusters found, skipping network activity plots.")
        return {}

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

    context_to_cluster = {}
    curr_idx = 0
    for i, gl in enumerate(group_labels):
        context_to_cluster[gl] = []
        for k in range(curr_idx, group_boundaries[i]):
            context_to_cluster[gl].append(sorted_order[k])
        curr_idx = group_boundaries[i]

    
    figsaxs = {}
    fig,axs,context_showed = plot_mean_cluster_activity(model,
                               loader,
                               cluster_to_neurons,
                               sorted_order,
                               context_to_cluster,
                               noiseless=noiseless,
                               num_samples=2)
    figsaxs[f'mean_cluster_activity_{context_showed}'] = (fig, axs)
    fig,axs = plot_global_cluster_correlation_with_context_blocks(model,
                                                        loader,
                                                        cluster_to_neurons,
                                                        context_to_cluster,
                                                        noiseless=noiseless,
                                                        max_batches=50,
                                                        show_cluster_ids=False)    
    figsaxs['global_cluster_correlation'] = (fig, axs)
    return figsaxs