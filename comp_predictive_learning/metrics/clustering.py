import torch 
from sklearn import metrics
# Clustering
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import copy 
import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def get_rnn_activities_and_sources_for_loader_for_clustering(model,loader,device='cuda'):
    
    rnn_cfg = copy.deepcopy(model.rnn_cfg)
    rnn_cfg.noise = 0.0
    noiseless_model = model.create_new_instance(new_params={'rnn_cfg': rnn_cfg}).to(DEVICE)
    activities = []
    contexts = []
    for i, batch in enumerate(loader):
        x,latents,context = batch
        _,_,activity,_ = noiseless_model(x)
        activities.append(activity.cpu())
        contexts.append(context.cpu())
    activities = torch.cat(activities,dim=1).permute(1,0,2).contiguous()
    contexts = torch.cat(contexts,dim=0).contiguous()
    return activities, contexts

@torch.no_grad()
def get_optimal_n_cluster(model,
                          activations=None,
                          contexts=None,
                          loader=None,
                          time_variance=True,
                          device='cuda',
                          max_num_clusters=25):

    assert (activations is not None) or (loader is not None), "Either activations or loader must be provided"
    if activations is None:
        activations, _,contexts = get_rnn_activities_and_sources_for_loader_for_clustering(model,loader,device=device)
    
    _,inv = contexts.unique(dim=0,return_inverse=True)
    rnn_activities_per_context = []
    for i in range(inv.max().item()+1):
        rnn_activities_per_context.append(activations[inv==i])
    
    rnn_activities_per_context = torch.stack(rnn_activities_per_context)
    rnn_activities_per_context = rnn_activities_per_context.permute(3,0,1,2)

    if time_variance:
        var_activities_per_context = rnn_activities_per_context.var(dim=3).mean(dim=2)
    else:
        var_activities_per_context = rnn_activities_per_context.var(dim=2).mean(dim=2)
        
    active_units = var_activities_per_context.sum(1) > 0.001
    var_activities_per_context = var_activities_per_context[active_units]
    norm_var_activities_per_context = (var_activities_per_context.T/(var_activities_per_context.max(1)[0]+1e-6)).T

    X = norm_var_activities_per_context.numpy()
    
    n_samples, n_features = X.shape
    if n_samples < 2 or n_features < 2:
        print("Not enough samples or features for clustering. Ensure that the input data has at least two samples and two features.")
        return 1, None, None, None,None

    max_num_clusters = min(max_num_clusters, n_samples)
    n_clusters = list(range(2, max_num_clusters))
    scores = list()

    max_num_clusters = min(max_num_clusters, n_samples)
    n_clusters = list(range(2, max_num_clusters))
    scores = list()
    for n_cluster in n_clusters:
        clustering = KMeans(n_cluster, algorithm='lloyd',random_state=0)
        clustering.fit(X) 
        labels = clustering.labels_ 
        score = metrics.silhouette_score(X, labels)
        scores.append(score)

    scores = np.array(scores)
    labels = KMeans(n_clusters[np.argmax(scores)], random_state=0).fit_predict(X)
    return n_clusters[np.argmax(scores)], scores, norm_var_activities_per_context, active_units,labels


def get_loss_per_context(model, loader):
    model.eval()
    loss_per_context = defaultdict(list)

    model.loss_fn = model.new_loss_fn(reduction="none")
        
    with torch.no_grad():
        for batch in loader:
            prim_in, latents, contexts = batch
            inputs, labels = prim_in, prim_in
            T,B, = prim_in.shape[:2]
            losses = model.loss(inputs)[0]
            losses = torch.mean(losses,dim=(0,2,3,4))
            for sample_idx, ctxt in enumerate(contexts):
                ctx_vec = tuple(ctxt.cpu().tolist())
                loss_per_context[ctx_vec].append(losses[sample_idx].item())

    mean_loss_per_context = {
        ctx: sum(vals)/len(vals)
        for ctx, vals in loss_per_context.items()
    }

    return mean_loss_per_context

def computer_lesioned_cluster_losses(model,
                                 dataloader,
                                 active_units,
                                 labels,
                                 number_of_clusters):
    mask = active_units.cpu().numpy()
    total_neurons = mask.shape[0]

    full_labels = np.full(total_neurons, -1, dtype=int)

    active_idx = np.where(mask)[0]

    full_labels[active_idx] = labels

    clusters_to_neurons = {
        c: active_idx[labels == c]
        for c in range(number_of_clusters)
    }

    original_network_losses = get_loss_per_context(model.create_new_instance().to(DEVICE), dataloader)
    cluster_losses = {}
    for clus in clusters_to_neurons:
        neurons_to_inhib = clusters_to_neurons[clus]
        lesioned_network = model.create_lesioned_instance(neurons_to_inhib).to(DEVICE)
        lesioned_network_losses = get_loss_per_context(lesioned_network, dataloader)
        cluster_losses[clus] = lesioned_network_losses
        print(f"Cluster {clus}, original loss: {np.mean(list(original_network_losses.values())):.4f}, new loss: {np.mean(list(lesioned_network_losses.values())):.4f}")
    return cluster_losses, original_network_losses,clusters_to_neurons


def plot_clusters(norm_var_activities_per_context,
                contexts_unique,
                labels,
                fig,
                ax):

    neuron_order  = np.argsort(labels)
    ctx           = contexts_unique.cpu().numpy()
    
    ctx_order     = np.lexsort((ctx[:,2], ctx[:,0]))            

    heat          = norm_var_activities_per_context[neuron_order, :].T                        # (n_ctx, n_neurons)
    heat_sorted   = heat[ctx_order, :]
    cluster_vals  = labels[neuron_order]

    boundaries    = np.where(np.diff(cluster_vals) != 0)[0] + 0.5

    centers       = []
    cluster_ids   = np.unique(cluster_vals)
    for k in cluster_ids:
        idxs = np.where(cluster_vals == k)[0]
        centers.append(idxs.mean())

    im = ax.imshow(heat_sorted, aspect='auto', origin='upper')

    ax.set_xticks(boundaries, minor=True)
    ax.tick_params(axis='x', which='minor', length=12, color='k')

    ax.set_xticks(centers, minor=False)
    ax.set_xticklabels([str(int(k)) for k in cluster_ids], fontsize=16)
    ax.tick_params(axis='x', which='major', length=0)

    n_ctx = heat_sorted.shape[0]
    ax.set_yticks(np.arange(n_ctx))
    
    ctx_vals = [ctx[ctord] for ctord in ctx_order]

    ax.set_yticklabels([f'{[int(c) for c in cs]}' for cs in ctx_vals],fontsize=8)

    ax.set_xlabel("Neuron clusters", fontsize=12)
    ax.set_ylabel("Contexts" +f"{[f'C{cidx}' for cidx in range(contexts_unique.shape[1])]}",fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized variance",fontsize=28)

    fig.set_tight_layout(True)
    return fig, ax


def create_context_activity_map(cluster_profiles,
                                contexts_unique,
                                informative_indices,
                                activity_threshold=0.5):
    """
    Creates a map from every context to all informative clusters active above a threshold.
    """
    if isinstance(contexts_unique, torch.Tensor):
        contexts_unique = contexts_unique.cpu().numpy()

    context_activity_map = {tuple(ctx): [] for ctx in contexts_unique}

    for cluster_id in informative_indices:
        profile = cluster_profiles[cluster_id]
        
        active_context_indices = np.where(profile > activity_threshold)[0]
        
        for ctx_idx in active_context_indices:
            context_tuple = tuple(contexts_unique[ctx_idx])
            context_activity_map[context_tuple].append(int(cluster_id))
            
    return context_activity_map
def analyze_and_sort_clusters(norm_var_activities_per_context,
                              contexts_unique,
                              labels,
                              selectivity_threshold=0.5,
                              purity_threshold=0.90):

    if isinstance(norm_var_activities_per_context, torch.Tensor):
        norm_var_activities_per_context = norm_var_activities_per_context.cpu().numpy()
    if isinstance(contexts_unique, torch.Tensor):
        contexts_unique = contexts_unique.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    flat_labels = labels.flatten()
    n_clusters = len(np.unique(flat_labels))
    n_contexts = norm_var_activities_per_context.shape[1]
    cluster_profiles = np.zeros((n_clusters, n_contexts))
    for i in range(n_clusters):
        neurons_in_cluster = np.where(flat_labels == i)[0]
        if len(neurons_in_cluster) > 0:
            cluster_profiles[i, :] = norm_var_activities_per_context[neurons_in_cluster, :].mean(axis=0)

    peak_activity = cluster_profiles.max(axis=1)
    sum_activity = cluster_profiles.sum(axis=1)
    mean_other_activity = (sum_activity - peak_activity) / (n_contexts - 1)
    epsilon = 1e-9
    selectivity_scores = (peak_activity - mean_other_activity) / (peak_activity + mean_other_activity + epsilon)

    informative_indices = np.where(selectivity_scores > selectivity_threshold)[0]
    non_selective_indices = np.where(selectivity_scores <= selectivity_threshold)[0]
    non_selective_indices.sort()

    context_to_peak_clusters_map = {}
    if len(informative_indices) > 0:
        informative_profiles = cluster_profiles[informative_indices]
        peak_context_indices = np.argmax(informative_profiles, axis=1)
        for i, cluster_id in enumerate(informative_indices):
            peak_ctx_idx = peak_context_indices[i]
            peak_ctx_vector = contexts_unique[peak_ctx_idx]
            peak_ctx_tuple = tuple(peak_ctx_vector)
            if peak_ctx_tuple not in context_to_peak_clusters_map:
                context_to_peak_clusters_map[peak_ctx_tuple] = []
            context_to_peak_clusters_map[peak_ctx_tuple].append(int(cluster_id))

    sorted_informative_clusters = []
    group_boundaries_by_cluster_count = []
    group_labels = []
    if len(informative_indices) > 0:
        unassigned_mask = np.ones(len(informative_indices), dtype=bool)
        ctx_order_y = np.lexsort(tuple(contexts_unique[:, i] for i in range(contexts_unique.shape[1] - 1, -1, -1)))
        y_pos = np.empty_like(ctx_order_y); y_pos[ctx_order_y] = np.arange(len(ctx_order_y))
        for dim in range(contexts_unique.shape[1]):
            current_indices_map = np.where(unassigned_mask)[0]
            if len(current_indices_map) == 0: break
            current_profiles = cluster_profiles[informative_indices[current_indices_map]]
            dim_vals = contexts_unique[:, dim]
            unique_dim_vals = np.unique(dim_vals)
            is_pure_for_dim = []
            for profile in current_profiles:
                total_variance = np.var(profile)
                if total_variance < epsilon:
                    is_pure_for_dim.append(False); continue
                residual_variance = np.mean([np.var(profile[dim_vals == v]) for v in unique_dim_vals])
                variance_explained = 1 - (residual_variance / total_variance)
                is_pure_for_dim.append(variance_explained > purity_threshold)
            pure_mask = np.array(is_pure_for_dim)
            if np.any(pure_mask):
                pure_indices_original = informative_indices[current_indices_map[pure_mask]]
                pure_profiles = cluster_profiles[pure_indices_original]
                sort_key1 = [unique_dim_vals[np.argmax([p[dim_vals == v].sum() for v in unique_dim_vals])] for p in pure_profiles]
                sort_key2 = y_pos[np.argmax(pure_profiles, axis=1)]
                sorted_indices = np.lexsort((sort_key2, sort_key1))
                sorted_informative_clusters.extend(pure_indices_original[sorted_indices])
                group_boundaries_by_cluster_count.append(len(sorted_informative_clusters))
                group_labels.append(f"C{dim}")
                unassigned_mask[current_indices_map[pure_mask]] = False
        mixed_indices_map = np.where(unassigned_mask)[0]
        
        if len(mixed_indices_map) > 0:
            mixed_indices_original = informative_indices[mixed_indices_map]
            non_selective_indices = list(non_selective_indices) + list(mixed_indices_original)


    final_cluster_order = sorted_informative_clusters + list(non_selective_indices)
    if len(non_selective_indices) > 0:
        group_boundaries_by_cluster_count.append(len(final_cluster_order))
        group_labels.append("Non-Selective")

    return (final_cluster_order, group_labels, group_boundaries_by_cluster_count,
            context_to_peak_clusters_map, informative_indices, cluster_profiles)
    


def plot_sequentially_sorted_clusters(norm_var_activities_per_context,
                                      contexts_unique,
                                      labels,
                                      final_cluster_order,
                                      group_labels,
                                      group_boundaries_by_cluster_count,
                                      fig,
                                      ax,
                                      put_y_label=True,
                                      group_name_map=None):

    if isinstance(norm_var_activities_per_context, torch.Tensor):
        norm_var_activities_per_context = norm_var_activities_per_context.cpu().numpy()
    if isinstance(contexts_unique, torch.Tensor):
        contexts_unique = contexts_unique.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    flat_labels = labels.flatten()

    neurons_to_plot = [n for c_idx in final_cluster_order for n in np.where(flat_labels == c_idx)[0]]
    neuron_order = np.array(neurons_to_plot)
    ctx_order_y = np.lexsort(tuple(contexts_unique[:, i] for i in range(contexts_unique.shape[1] - 1, -1, -1)))
    heat = norm_var_activities_per_context[neuron_order, :].T
    heat_sorted = heat[ctx_order_y, :]
    
    im = ax.imshow(heat_sorted, aspect='auto', origin='upper', cmap='viridis')

    neurons_per_cluster = {i: np.sum(flat_labels == i) for i in np.unique(flat_labels)}
    group_neuron_counts = []
    start_cluster_idx = 0
    for end_cluster_idx in group_boundaries_by_cluster_count:
        clusters_in_group = final_cluster_order[start_cluster_idx:end_cluster_idx]
        group_neuron_counts.append(sum(neurons_per_cluster.get(c, 0) for c in clusters_in_group))
        start_cluster_idx = end_cluster_idx
    neuron_group_boundaries = np.cumsum(group_neuron_counts)

    for boundary in neuron_group_boundaries[:-1]:
        ax.axvline(boundary - 0.5, color='cyan', linestyle='-', linewidth=2.5)

    line_colors = ['white', 'yellow', 'magenta', 'orange', 'red']
    x_start = -0.5 

    for i, (label, boundary) in enumerate(zip(group_labels, neuron_group_boundaries)):
        x_end = boundary - 0.5
        color = line_colors[i % len(line_colors)]
        
        if not label.startswith("C"):
            continue
        dim_to_group_by = int(label.replace("C", ""))
        
        sorted_dim_contexts = contexts_unique[ctx_order_y, dim_to_group_by]

        diffs = np.diff(sorted_dim_contexts)
        
        all_boundaries = np.where(diffs != 0)[0] + 0.5

        cycle_starts = np.where(diffs < 0)[0] + 0.5

        normal_boundaries = np.setdiff1d(all_boundaries, cycle_starts)

        ax.hlines(normal_boundaries, xmin=x_start, xmax=x_end,
                color=color, linestyle='--' , linewidth=1.5, alpha=0.9)

        ax.hlines(cycle_starts, xmin=x_start, xmax=x_end,
                color=color, linestyle='-', linewidth=1.5, alpha=0.9)

        x_start = x_end

    neuron_counts_sorted = [neurons_per_cluster.get(c, 0) for c in final_cluster_order]
    cluster_boundaries_by_neuron = np.cumsum(neuron_counts_sorted)
    centers = cluster_boundaries_by_neuron - (np.array(neuron_counts_sorted) / 2)
    ax.set_xticks(centers)
    ax.set_xticklabels([str(k) for k in final_cluster_order], fontsize=16)
    ax.tick_params(axis='x', which='major', length=0)
    minor_tick_locations = cluster_boundaries_by_neuron[:-1] - 0.5
    ax.set_xticks(minor_tick_locations, minor=True)
    ax.tick_params(axis='x', which='minor', length=6, color='black')
    ax.set_yticks(np.arange(len(contexts_unique)))
    ctx_vals_sorted = [contexts_unique[ctord] for ctord in ctx_order_y]
    varying_dims = [i for i in range(contexts_unique.shape[1]) if len(np.unique(contexts_unique[:, i])) > 1]
    if not varying_dims: varying_dims = list(range(contexts_unique.shape[1]))
    
    if put_y_label:
        new_yticklabels = [f'{[int(cs[i]) for i in varying_dims]}' for cs in ctx_vals_sorted]
        ax.set_yticklabels(new_yticklabels, fontsize=8)
        ax.set_ylabel("Contexts", fontsize=26)
    else:
        ax.set_yticklabels([])
        ax.set_yticks([])
    group_sizes = np.diff(np.concatenate(([0], neuron_group_boundaries)))
    group_starts = np.concatenate(([0], neuron_group_boundaries[:-1]))
    group_centers = group_starts + group_sizes / 2.0

    group_labels_mapped = [group_name_map.get(lbl, lbl) if group_name_map else lbl
                           for lbl in group_labels]

    for xc, lbl in zip(group_centers, group_labels_mapped):
        ax.annotate(lbl.replace("Non-Selective", "NS"),
                    xy=(xc, -0.04), xycoords=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=20)

    ax.set_xlabel("Neuron Clusters", fontsize=26, labelpad=30)
    fig.subplots_adjust(bottom=0.22)  # tweak as needed

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized variance",fontsize=24)
    fig.set_tight_layout(True)
    
    return fig, ax


def plot_lesion_effects_across_clusters(model,
                                        loader,
                                        cluster_to_neurons,
                                        t_start=1,
                                        t_end=4,
                                        num_samples=4,
                                        DEVICE='cuda'):

    inputs, _, contexts = next(iter(loader))
    inputs, contexts = inputs.to(DEVICE), contexts.to(DEVICE)

    figs,axs = [], []
    def keep_axis_hide_ticks(ax_):
        ax_.set_xticks([])
        ax_.set_yticks([])
        for s in ax_.spines.values():
            s.set_visible(False)

    for b in range(num_samples):
        sample_input  = inputs[:, b:b+1]
        sample_context = contexts[b]

        with torch.no_grad():
            baseline_recons = model(sample_input)[0].detach()

        num_clusters = len(cluster_to_neurons)
        nrows = 2 + num_clusters
        fig, ax = plt.subplots(
            nrows, t_end - t_start,
            figsize=((t_end - t_start) * 3, nrows * 3),
            constrained_layout=False  
        )
        ax = np.atleast_2d(ax)
        fig.subplots_adjust(left=0.18, top=0.97, hspace=0.1, wspace=0.1) 

        ax[0, 0].set_ylabel("Ground Truth", fontsize=12, weight='bold', labelpad=12)
        ax[1, 0].set_ylabel("Original\n Reconstruction", fontsize=12, weight='bold', labelpad=12)

        for t_idx, t in enumerate(range(t_start, t_end)):
            ax[0, t_idx].imshow(sample_input[t + 1, 0].cpu().permute(1, 2, 0).numpy())

            out = baseline_recons[t, 0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()

            ax[1, t_idx].imshow(out)

            if t_idx == 0:
                keep_axis_hide_ticks(ax[0, t_idx])
                keep_axis_hide_ticks(ax[1, t_idx])
            else:
                ax[0, t_idx].axis("off")
                ax[1, t_idx].axis("off")

            ax[-1, t_idx].set_xlabel(f"t={t+1}", fontsize=12)

        for i, cluster_id in enumerate(sorted(cluster_to_neurons.keys())):
            row = i + 2
            ax[row, 0].set_ylabel(f"Lesion C{cluster_id}", fontsize=12, weight='bold', labelpad=12)

            new_network = model.create_lesioned_instance(cluster_to_neurons[cluster_id]).to(DEVICE)
            with torch.no_grad():
                inhib_recons = new_network(sample_input)[0].detach()

            for t_idx, t in enumerate(range(t_start, t_end)):

                out = inhib_recons[t, 0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()

                ax[row, t_idx].imshow(out)
                keep_axis_hide_ticks(ax[row, t_idx]) 

        figs.append(fig)
        axs.append(ax)
    return figs, axs,[int(c) for c in sample_context[:3]]


def plot_sorted_cluster(config,
                        model,
                        loader,
                        fig=None,
                        ax=None,
                        selectivity_threshold=0.1,
                        purity_threshold=0.5):
    activations, contexts = get_rnn_activities_and_sources_for_loader_for_clustering(model, loader)
    max_num_clusters, scores, norm_var_activities_per_context, active_units, labels = get_optimal_n_cluster(model, activations=activations, contexts=contexts, time_variance=False, device=DEVICE)
    
    if max_num_clusters ==1:
        print("Only one cluster found, skipping plotting.")
        return None,None
    
    unique_contexts, inv = contexts.unique(dim=0, return_inverse=True)

    print(f"Optimal number of clusters: {max_num_clusters}, Silhouette scores: {scores}")

    (sorted_order, group_labels, group_boundaries,
    peak_map, informative_indices, cluster_profiles) = analyze_and_sort_clusters(
        norm_var_activities_per_context=norm_var_activities_per_context.cpu().numpy(),
        contexts_unique=unique_contexts.cpu(),
        labels=labels,
        selectivity_threshold= selectivity_threshold,
        purity_threshold= purity_threshold
    )
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
    plot_sequentially_sorted_clusters(
        norm_var_activities_per_context=norm_var_activities_per_context.cpu().numpy(),
        contexts_unique=unique_contexts.cpu(),
        labels=labels,
        final_cluster_order=sorted_order,
        group_labels=group_labels,
        group_boundaries_by_cluster_count=group_boundaries,
        put_y_label= "auto" in config.model.type,
        fig=fig,
        ax=ax
    )

    return fig,ax 

def plot_cluster_and_lesions(config,model,loader,fig=None, ax=None):
    
    activations, contexts = get_rnn_activities_and_sources_for_loader_for_clustering(model, loader)
    max_num_clusters, scores, norm_var_activities_per_context, active_units, labels = get_optimal_n_cluster(model, activations=activations, contexts=contexts, time_variance=False, device=DEVICE)
    unique_contexts, inv = contexts.unique(dim=0, return_inverse=True)

    if max_num_clusters == 1:
        print("Only one cluster found, skipping plotting.")
        return None,None,None,None
    print(f"Optimal number of clusters: {max_num_clusters}, Silhouette scores: {scores}")

    (sorted_order, group_labels, group_boundaries,
    peak_map, informative_indices, cluster_profiles) = analyze_and_sort_clusters(
        norm_var_activities_per_context=norm_var_activities_per_context.cpu().numpy(),
        contexts_unique=unique_contexts.cpu(),
        labels=labels,
        selectivity_threshold= 0.2,
        purity_threshold= 0.5
    )
    fig_cluster,ax_cluster = plt.subplots(1, 1, figsize=(10, 8))
    
    plot_sequentially_sorted_clusters(
        norm_var_activities_per_context=norm_var_activities_per_context.cpu().numpy(),
        contexts_unique=unique_contexts.cpu(),
        labels=labels,
        final_cluster_order=sorted_order,
        group_labels=group_labels,
        group_boundaries_by_cluster_count=group_boundaries,
        group_name_map={"C0":"shape","C1":"color","C2":"position"},
        put_y_label= not "auto" in config.model.type,
        fig=fig_cluster,
        ax=ax_cluster
    )

    fig_lesion,ax_lesion = plt.subplots(1, 1, figsize=(10, 8))
    cluster_losses, original_network_losses,cluster_to_neurons = computer_lesioned_cluster_losses(model,
                                                                           dataloader=loader,
                                                                           active_units=active_units,
                                                                           labels=labels,
                                                                           number_of_clusters=max_num_clusters)

    plot_cluster_lesion_delta(
        original_network_losses=original_network_losses,
        lesioned_cluster_loss=cluster_losses,
        cluster_order=sorted_order, 
        put_y_label= not "auto" in config.model.type,
        fig=fig_lesion,
        ax=ax_lesion)
    
    figs,axs,context_for_lesion = plot_lesion_effects_across_clusters(
    model=model,
    loader=loader,
    cluster_to_neurons=cluster_to_neurons,
    t_start=1, 
    t_end = 3,  
    num_samples=1, 
    DEVICE=DEVICE)
        
    return fig_cluster,ax_cluster,fig_lesion,ax_lesion,figs,axs,context_for_lesion
    

def plot_cluster_lesion_delta(original_network_losses,
                              lesioned_cluster_loss,
                              fig,
                              ax,
                              put_y_label=True,
                              cluster_order=None):


    
    contexts = sorted(original_network_losses.keys(), key=lambda x: tuple((x[i] for i in range(len(x)))))
    clusters = sorted(lesioned_cluster_loss.keys())

    if cluster_order is not None:
        clusters = [c for c in cluster_order if c in lesioned_cluster_loss]
    else:
        clusters = sorted(lesioned_cluster_loss.keys())

    diff = np.zeros((len(contexts), len(clusters)))
    for i, ctx in enumerate(contexts):
        orig = original_network_losses[ctx]
        for j, clus in enumerate(clusters):
            new = lesioned_cluster_loss[clus][ctx]
            diff[i, j] = new - orig

    ax.set_xticks(np.arange(len(clusters)))
    ax.set_xticklabels([str(c) for c in clusters], fontsize=16)
    ax.set_xlabel("Lesioned cluster",fontsize=26)
    
    contexts_array = np.array(contexts)
        
    n_dims = contexts_array.shape[1]
    varying_dims = [i for i in range(n_dims) if len(np.unique(contexts_array[:, i])) > 1]

    ctx_labels = []
    for cs in contexts:
        varying_cs = [int(cs[i]) for i in varying_dims]
        ctx_labels.append(f'{varying_cs}')
    
    if put_y_label:
        ax.set_yticks(np.arange(len(contexts)))
        ax.set_yticklabels(ctx_labels, fontsize=8)
        ax.set_ylabel("Contexts",fontsize=26)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    im = ax.imshow(diff, aspect='auto', origin='upper', cmap='binary')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(" Lesioned Loss − Original Loss",fontsize=24)

    fig.set_tight_layout(True)
    return fig, ax
    
    