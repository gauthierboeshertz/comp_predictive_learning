from collections import defaultdict
import copy
from matplotlib import cm
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from compo_predictive_learning.metrics.clustering import analyze_and_sort_clusters, get_optimal_n_clusters, get_rnn_activities_and_sources_for_loader_for_clustering

# -----------------------------
# Shared styling
# -----------------------------
DEFAULT_STYLE = {
    "title_fs": 20,
    "label_fs": 18,
    "tick_fs": 16,
    "legend_fs": 16,
    "cbar_label_fs": 15,
    "cbar_tick_fs": 14,
    "line_w": 2.2,
    "grid_alpha": 0.25,
    "heatmap_tick_fs": 14,
    "panel_size": (7.5, 6.0),
    "title_pad": 15,
}

def _style(style=None):
    s = DEFAULT_STYLE.copy()
    if style is not None:
        s.update(style)
    return s


# -----------------------------
# One shared metadata pass
# -----------------------------
def compute_cluster_metadata(model, loader, activations=None, contexts=None):
    
    if activations is None or contexts is None:
        activations, contexts = get_rnn_activities_and_sources_for_loader_for_clustering(model, loader)

    max_num_clusters, scores, norm_var_activities_per_context, active_units, labels = get_optimal_n_clusters(
        model,
        activations=activations,
        contexts=contexts,
        time_variance=False
    )

    if max_num_clusters <= 2:
        raise ValueError("Not enough clusters found, skipping composite plot.")

    unique_contexts, inv = contexts.unique(dim=0, return_inverse=True)

    (
        sorted_order,
        group_labels,
        group_boundaries,
        peak_map,
        informative_indices,
        cluster_profiles
    ) = analyze_and_sort_clusters(
        norm_var=norm_var_activities_per_context.cpu().numpy(),
        contexts_unique=unique_contexts.cpu(),
        labels=labels,
        selectivity_threshold=0.2,
        purity_threshold=0.5
    )

    original_indices = np.where(active_units)[0]
    cluster_to_neurons = {}
    for cluster_id in np.unique(labels):
        neurons_in_cluster_mask = labels == cluster_id
        cluster_to_neurons[cluster_id] = original_indices[neurons_in_cluster_mask]

    context_to_cluster = {}
    curr_idx = 0
    for i, gl in enumerate(group_labels):
        context_to_cluster[gl] = []
        for k in range(curr_idx, group_boundaries[i]):
            context_to_cluster[gl].append(sorted_order[k])
        curr_idx = group_boundaries[i]

    return {
        "activations": activations,
        "contexts": contexts,
        "unique_contexts": unique_contexts,
        "sorted_order": sorted_order,
        "group_labels": group_labels,
        "group_boundaries": group_boundaries,
        "cluster_to_neurons": cluster_to_neurons,
        "context_to_cluster": context_to_cluster,
        "labels": labels,
        "active_units": active_units,
        "norm_var_activities_per_context": norm_var_activities_per_context,
    }


# -----------------------------
# 1) Mean cluster activity -> one panel
# -----------------------------

@torch.no_grad()
def plot_mean_cluster_activity_on_ax(
    model,
    loader,
    cluster_to_neurons,
    cluster_order,
    context_to_cluster,
    ax,
    noiseless=True,
    context_idx=0,
    device="cuda",
    style=None,
    group_id_to_name={"C0": "Shape", "C1": "Colour", "C2": "Position"},
    title="Mean cluster activity",
):
    s = _style(style)

    unique_context_groups = sorted(context_to_cluster.keys())
    base_colors = sns.color_palette("husl", n_colors=len(unique_context_groups))
    context_group_to_base_color = {
        group_id: color for group_id, color in zip(unique_context_groups, base_colors)
    }
    cluster_to_group_map = {
        cid: gid for gid, cids in context_to_cluster.items() for cid in cids
    }

    valid_cluster_order = [cid for cid in cluster_order if cid in cluster_to_neurons]

    if noiseless:
        rnn_cfg = copy.deepcopy(model.rnn_cfg)
        rnn_cfg.noise = 0.0
        plot_model = model.create_new_instance(new_params={"rnn_cfg": rnn_cfg}).to(device)
    else:
        plot_model = model

    chosen_context = None

    for b, (inputs, _, contexts) in enumerate(loader):
        if b < context_idx:
            continue
        if b > context_idx:
            break

        inputs = inputs.to(device)
        _, _, activities, _ = plot_model(inputs)   # (T, B, N)
        batch_mean_activity = activities.mean(dim=1).detach().cpu().numpy()  # (T, N)

        cluster_mean_traces = {}
        cluster_activeness = {}

        for cluster_id in valid_cluster_order:
            neuron_indices = cluster_to_neurons.get(cluster_id, [])
            if len(neuron_indices) > 0:
                mean_trace = batch_mean_activity[:, neuron_indices].mean(axis=1)
                cluster_mean_traces[cluster_id] = mean_trace
                cluster_activeness[cluster_id] = mean_trace.mean()

        most_active_cluster_per_group = {}
        print(context_to_cluster)
        for group_id, cluster_ids_in_group in context_to_cluster.items():
            valid_clusters_in_group = [
                cid for cid in cluster_ids_in_group if cid in cluster_activeness
            ]
            if not valid_clusters_in_group:
                continue
            most_active_id = max(valid_clusters_in_group, key=lambda cid: cluster_activeness[cid])
            most_active_cluster_per_group[group_id] = most_active_id

        for cluster_id in valid_cluster_order:
            if cluster_id not in cluster_mean_traces:
                continue

            mean_trace = cluster_mean_traces[cluster_id]
            group_id = cluster_to_group_map.get(cluster_id, None)
            is_most_active = (most_active_cluster_per_group.get(group_id) == cluster_id)

            color = "gray"
            if group_id in context_group_to_base_color:
                base_color = context_group_to_base_color[group_id]
                color = base_color if is_most_active else sns.light_palette(base_color, n_colors=5)[1]

            label = None
            if is_most_active and group_id in group_id_to_name:
                label = f"Cluster {cluster_id} (Moves {group_id_to_name[group_id]} by {context_to_cluster[group_id].index(cluster_id)})" if group_id in unique_context_groups else f"Cluster {cluster_id}"
            elif is_most_active:
                label = f"Cluster {cluster_id}"
            ax.plot(
                mean_trace,
                linewidth=s["line_w"],
                color=color,
                label=label
            )

        chosen_context = [int(c) for c in contexts[0, :3]]
        break

    if chosen_context is not None:
        context_names = "(" + ", ".join(
            group_id_to_name.get(f"C{i}", f"C{i}")
            for i in range(len(chosen_context))
        ) + ")"

        title_text = (
            f"{title}\n"
            f"context={tuple(chosen_context)}\n"
            f"{context_names}"
        )
    else:
        title_text = title

    ax.set_title(
        title_text,
        fontsize=s["title_fs"],
        fontweight="bold",
        pad=s["title_pad"]
    )
    ax.set_xlabel("Time step", fontsize=s["label_fs"])
    ax.set_ylabel("Mean activity", fontsize=s["label_fs"])
    ax.tick_params(axis="both", which="major", labelsize=s["tick_fs"])
    ax.grid(True, linestyle="--", alpha=s["grid_alpha"])

    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(loc="lower right", fontsize=s["legend_fs"], frameon=True)

    return ax


# -----------------------------
# 2) Active clusters over time -> one panel
# FIXED: averages over neurons only, not over batch
# -----------------------------
@torch.no_grad()
def plot_num_active_clusters_over_time_on_ax(
    dataloader,
    model,
    context_to_cluster,
    cluster_to_neurons,
    ax,
    device="cuda",
    threshold_ratio=0.2,
    start_from_second_activity=False,
    variance_mode="std",    # "std", "sem", or "var"
    latent_names=("Shape", "Colour", "Position"),
    style=None,
    title="Number of active clusters over time",
):
    s = _style(style)

    model.eval()
    latent_keys = list(context_to_cluster.keys())
    n_populations = 4
    colors = sns.color_palette("husl", n_colors=n_populations)
    counts_per_population = [[] for _ in range(n_populations)]

    for batch in dataloader:
        inputs, latents, context = batch
        inputs = inputs.to(device)

        # (T, B, H) -> (B, T, H)
        rnn_act = model(inputs)[2].permute(1, 0, 2)

        if start_from_second_activity:
            rnn_act = rnn_act[:, 1:, :]

        for latent_idx in range(n_populations):
            latent_key = latent_keys[latent_idx]
            cluster_ids = context_to_cluster[latent_key]

            cluster_means = []
            for cluster_id in cluster_ids:
                neuron_indices = torch.as_tensor(
                    cluster_to_neurons[cluster_id],
                    device=device,
                    dtype=torch.long
                )

                acts = rnn_act[:, :, neuron_indices] 

                cluster_mean = acts.mean(dim=(0, -1)).unsqueeze(0)

                cluster_means.append(cluster_mean)

            cluster_means = torch.stack(cluster_means, dim=-1)

            max_mean = cluster_means.max(dim=-1, keepdim=True).values      # (B, T, 1)
            active_mask = cluster_means > (threshold_ratio * max_mean)
            n_active = active_mask.sum(dim=-1)                              # (B, T)

            counts_per_population[latent_idx].append(n_active.cpu().numpy())

    summary = {}
    for latent_idx in range(n_populations):
        latent_key = latent_keys[latent_idx]
        counts = np.concatenate(counts_per_population[latent_idx], axis=0)  # (N, T)

        mean_counts = counts.mean(axis=0)
        std_counts  = counts.std(axis=0)
        var_counts  = counts.var(axis=0)
        sem_counts  = std_counts / np.sqrt(max(counts.shape[0], 1))

        if variance_mode == "std":
            spread = std_counts
        elif variance_mode == "sem":
            spread = sem_counts
        elif variance_mode == "var":
            spread = var_counts
        else:
            raise ValueError("variance_mode must be one of: 'std', 'sem', 'var'")

        x = np.arange(mean_counts.shape[0])

        ax.plot(
            x,
            mean_counts,
            label=latent_names[latent_idx] if latent_idx < len(latent_names) else str(latent_key),
            color=colors[latent_idx],
            linewidth=s["line_w"]
        )
        ax.fill_between(
            x,
            mean_counts - spread,
            mean_counts + spread,
            alpha=0.12,
            color=colors[latent_idx]
        )

        summary[latent_key] = {
            "counts": counts,
            "mean": mean_counts,
            "std": std_counts,
            "var": var_counts,
            "sem": sem_counts,
        }

    ax.set_title(title, fontsize=s["title_fs"], fontweight="bold",pad=s["title_pad"])
    ax.set_xlabel("Time Step", fontsize=s["label_fs"])
    ax.set_ylabel("Number of active clusters", fontsize=s["label_fs"])
    ax.tick_params(axis="both", which="major", labelsize=s["tick_fs"])
    ax.grid(True, alpha=s["grid_alpha"])
    ax.legend(fontsize=s["legend_fs"], frameon=True)

    return summary

@torch.no_grad()
def decompose_rnn_dynamics_by_cluster(rnn_layer, rnn_inputs, cluster_to_neurons):
    """
    rnn_inputs: (T, B, input_dim)
    cluster_to_neurons: dict mapping cluster_id -> list of neuron indices
    """
    T, B, _ = rnn_inputs.shape
    device = rnn_inputs.device

    state = torch.zeros(B, rnn_layer.hidden_dim, device=device)
    out_prev = torch.zeros_like(state)

    out = defaultdict(list)
    
    # Nested dictionary to store inputs: cluster_inputs[source_id][target_id]
    cluster_inputs = {
        src: {tgt: [] for tgt in cluster_to_neurons.keys()} 
        for src in cluster_to_neurons.keys()
    }

    # Extract the full hidden-to-hidden weight matrix
    # Assumes rnn_layer.weight_hh is an nn.Linear module
    W_hh = rnn_layer.weight_hh.weight 

    for t in range(T):
        # 1. Standard Dynamics
        I_in = rnn_layer.alpha * rnn_layer.input_layer(rnn_inputs[t])
        I_rec = rnn_layer.alpha * rnn_layer.weight_hh(out_prev)
        mem_leak = (1-rnn_layer.alpha) * state
        net = I_in + I_rec + mem_leak
        net_without_input = I_rec + mem_leak
        new_state = net
        new_out = rnn_layer.activation(new_state)

        for src_id, src_idx in cluster_to_neurons.items():
            # Source activity at t-1. Shape: (B, len(src_idx))
            src_activity = out_prev[:, src_idx] 
            
            for tgt_id, tgt_idx in cluster_to_neurons.items():
                # Sub-matrix of weights from Source -> Target
                # Shape: (len(tgt_idx), len(src_idx))
                W_sub = W_hh[tgt_idx][:, src_idx]  # Extract relevant weights for this source-target pair
                
                I_src_to_tgt = rnn_layer.alpha * (src_activity @ W_sub.T)
                
                cluster_inputs[src_id][tgt_id].append(I_src_to_tgt.clone())

        # 3. Store Data
        out["state_before"].append(state.clone())
        out["input_current"].append(I_in.clone())
        out["recurrent_current"].append(I_rec.clone())
        out["leak_current"].append(mem_leak.clone())
        out["net_current"].append(net.clone())
        out["net_without_input"].append(net_without_input.clone())
        out["state_after"].append(new_state.clone())
        out["activity"].append(new_out.clone())

        state = new_state
        out_prev = new_out

    # Stack the standard dynamics into tensors of shape (T, B, hidden_dim)
    stacked_out = {k: torch.stack(v, dim=0) for k, v in out.items()}
    
    # Stack the cluster inputs into tensors of shape (T, B, len(target_idx))
    stacked_cluster_inputs = {
        src: {
            tgt: torch.stack(cluster_inputs[src][tgt], dim=0) 
            for tgt in cluster_to_neurons.keys()
        }
        for src in cluster_to_neurons.keys()
    }
    
    # Append the cluster tracking to the output dictionary
    stacked_out["cluster_to_cluster_rec"] = stacked_cluster_inputs
    return stacked_out

def get_rnn_decomposed(
    latent_idx,
    dynamic_idx,
    dataloader,
    model,
    context_to_cluster,
    cluster_to_neurons,
    device='cuda'
):
    
    # --- 1. Neuron Selection Logic ---
    latent_keys = list(context_to_cluster.keys())
    target_latent_key = latent_keys[latent_idx]
    
    # Get cluster and neurons
    target_cluster_id = context_to_cluster[target_latent_key][dynamic_idx]
    neuron_indices = cluster_to_neurons[target_cluster_id]
    
    print(f"Analyzing Latent: '{target_latent_key}' (Idx {latent_idx}) | Dynamic: {dynamic_idx}")
    print(f"Cluster ID: {target_cluster_id} | Neurons: {len(neuron_indices)}")
    
    # --- 2. Data Collection ---
    collected_activities = []
    collected_latents = []
    
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, latents, context = batch
            latents = torch.cat([latents[:,:,:1], latents[:,:,2:]], dim=2)  # remove scale info from latents
            inputs = inputs.to(device)
            
            print(f"Batch context: {context[0].cpu().numpy()}")  # Debug print to check contexts
            batch_dynamic_indices = context[:, latent_idx]
            mask = (batch_dynamic_indices == dynamic_idx)
            
            if mask.sum() == 0:
                print("No sequences found for this dynamic in the current batch, skipping...")
                continue
            
            rnn_inputs = model.encoder(inputs)  # (T, B, input_dim)
            rnn_decomposed = decompose_rnn_dynamics_by_cluster(model.rnn, rnn_inputs, cluster_to_neurons)
            decomposed_by_latents_cluster = {}
            for cluster_id in context_to_cluster[target_latent_key]:
                cluster_neurons = cluster_to_neurons[cluster_id]
                decomposed_by_latents_cluster[cluster_id] = {
                    k: v[:, :, cluster_neurons] for k, v in rnn_decomposed.items() if k != "cluster_to_cluster_rec"
                }
            decomposed_by_latents_cluster["cluster_to_cluster_rec"] = rnn_decomposed["cluster_to_cluster_rec"]
            latents = latents.permute(1, 0, 2)  # (B, T, Latents)
            return decomposed_by_latents_cluster, context[mask][0].cpu().numpy()  # Return the first matching context for reference
    
def cosine_similarity(a, b):
    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return (a_norm * b_norm).sum(dim=-1).mean()

def plot_cosine_similarity_between_input_and_recurrent_input(model,
                                                             loader,
                                                             latent_idx,
                                                             dynamic_idx,
                                                             ax,
                                                            context_to_cluster,
                                                            cluster_to_neurons,
                                                            group_id_to_name={"C0": "Shape", "C1": "Colour", "C2": "Position"},
                                                            ):  
    s = _style()
    decomposed, context = get_rnn_decomposed(
        latent_idx=latent_idx,
        dynamic_idx=dynamic_idx,
        dataloader=loader,
        model=model,
        context_to_cluster=context_to_cluster,
        cluster_to_neurons=cluster_to_neurons)

    cos_between_input_and_state = {}
    num_negative_rec_input = {}
    activitit = {}
    for cluster_id, decomp in decomposed.items():
        if cluster_id == "cluster_to_cluster_rec":
            continue
        cos_between_input_and_state[cluster_id] = []
        num_negative_rec_input[cluster_id] = []
        activitit[cluster_id] = []
        for t in range(6):#decomp["input_current"].shape[0]
            mask = decomp["input_current"][t, :, :] > 0
            mask = torch.ones_like(mask, dtype=torch.bool)  # consider all neurons in the cluster, not just those with positive input
            I_in = decomp["input_current"][t, :, :][mask]#.reshape(-1)
            state = decomp["recurrent_current"][t, :, :][mask] #.reshape(-1) net_without_input,recurrent_current
            # self_rec =  decomposed["cluster_to_cluster_rec"][cluster_id][cluster_id][t, :, :][mask] 
            # self_rec = decomposed["cluster_to_cluster_rec"][cluster_id][cluster_id][t, :, :][mask]  + decomp["leak_current"][t, :, :][mask]
            cos = cosine_similarity(I_in, state)
            activitit[cluster_id].append(decomp["activity"][t, :, :][mask].mean().item())
            # cos = np.corrcoef(I_in.cpu().numpy(), self_rec.cpu().numpy())[0, 1]
            # cos = (I_in + state).mean()
            
            cos_between_input_and_state[cluster_id].append(cos.item())
            num_negative_rec_input[cluster_id].append((decomp["input_current"][t, :, :]).float().mean().item())

    fig = ax.figure
    colors = cm.hot(np.linspace(0, 1, len(cos_between_input_and_state)))
    dynamic_name = group_id_to_name.get(list(group_id_to_name.keys())[latent_idx])
    for cluster_id, cos_values in cos_between_input_and_state.items():
        ax.plot(cos_values, label=f"Cluster {cluster_id} (Moves {dynamic_name} by {context_to_cluster[list(group_id_to_name.keys())[latent_idx]].index(cluster_id)})", color=colors[list(cos_between_input_and_state.keys()).index(cluster_id)])
        # ax.plot(activitit[cluster_id], linestyle="--", color=colors[list(cos_between_input_and_state.keys()).index(cluster_id)])
    ax.set_xlabel("Time Step",fontsize=s["label_fs"])
    ax.grid(True, alpha=0.25)
    ax.set_facecolor("white")
    ax.set_ylabel("Cosine similarity",fontsize=s["label_fs"])
    ax.tick_params(axis='both', which='major', labelsize=s["tick_fs"])
    ax.set_title(f"Cosine similarity between  input and \n recurrent current for context {tuple([int(c) for c in context[:3]])}", fontsize=s["title_fs"],fontweight="bold", pad=s["title_pad"])
    ax.legend(fontsize=s["legend_fs"],loc="lower right", frameon=True)
    return ax

    

# def plot_context_weights_heatmaps_on_ax(
#     W,
#     all_neurons_groups,
#     context_to_cluster,
#     ax,
#     fig,
#     style=None,
#     title="Mean recurrent weights",
#     context_to_context_name={"C0": "Shape", "C1": "Colour", "C2": "Position"},
# ):
#     s = _style(style)

#     sets_int = {
#         str(k): np.array(list(map(int, v)), dtype=int)
#         for k, v in all_neurons_groups.items()
#     }

#     ordered_clusters = []
#     for ctx, clusters in context_to_cluster.items():
#         valid_clusters = [str(c) for c in clusters if str(c) in sets_int]
#         for cluster_id in valid_clusters:
#             ordered_clusters.append((ctx, cluster_id))

#     if not ordered_clusters:
#         raise ValueError("No valid clusters found to plot.")

#     n_clusters = len(ordered_clusters)

#     def block_weights(src_id, tgt_id):
#         src_neurons, tgt_neurons = sets_int[src_id], sets_int[tgt_id]
#         if src_neurons.size == 0 or tgt_neurons.size == 0:
#             return np.array([])
#         return W[np.ix_(src_neurons, tgt_neurons)].ravel()

#     heatmap_matrix = np.zeros((n_clusters, n_clusters))
#     for i in range(n_clusters):
#         _, src_id = ordered_clusters[i]
#         for j in range(n_clusters):
#             _, tgt_id = ordered_clusters[j]
#             weights = block_weights(src_id, tgt_id)
#             heatmap_matrix[i, j] = np.mean(weights) if weights.size > 0 else 0.0

#     v_max = np.max(np.abs(heatmap_matrix))
#     im = ax.imshow(
#         heatmap_matrix,
#         cmap="bwr",
#         vmin=-v_max,
#         vmax=v_max,
#         aspect="auto"
#     )

#     ax.set_xticks(np.arange(n_clusters))
#     ax.set_yticks(np.arange(n_clusters))
#     ax.set_xticklabels([cid for _, cid in ordered_clusters], rotation=90, fontsize=s["heatmap_tick_fs"])
#     ax.set_yticklabels([cid for _, cid in ordered_clusters], fontsize=s["heatmap_tick_fs"])

#     ax.set_title(title, fontsize=s["title_fs"], fontweight="bold", pad=s["title_pad"])
#     ax.set_xlabel("Target cluster", fontsize=s["label_fs"])
#     ax.set_ylabel("Source cluster", fontsize=s["label_fs"])
#     ax.tick_params(axis="both", which="major", labelsize=s["heatmap_tick_fs"])

#     boundaries = [
#         i for i in range(n_clusters - 1)
#         if ordered_clusters[i][0] != ordered_clusters[i + 1][0]
#     ]
#     for b in boundaries:
#         ax.axhline(b + 0.5, linewidth=1.5, color="black")
#         ax.axvline(b + 0.5, linewidth=1.5, color="black")

#     cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     cbar.set_label("Mean weight", fontsize=s["cbar_label_fs"])
#     cbar.ax.tick_params(labelsize=s["cbar_tick_fs"])

#     return {
#         "heatmap_matrix": heatmap_matrix,
#         "ordered_clusters": ordered_clusters,
#         "image": im
#     }

def plot_context_weights_heatmaps_on_ax(
    W,
    all_neurons_groups,
    context_to_cluster,
    ax,
    fig,
    style=None,
    title="Mean recurrent weights",
    context_to_context_name={"C0": "Shape", "C1": "Colour", "C2": "Position"},
):
    s = _style(style)

    sets_int = {
        str(k): np.array(list(map(int, v)), dtype=int)
        for k, v in all_neurons_groups.items()
    }

    ordered_clusters = []
    for ctx, clusters in context_to_cluster.items():
        valid_clusters = [str(c) for c in clusters if str(c) in sets_int]
        for cluster_id in valid_clusters:
            ordered_clusters.append((ctx, cluster_id))

    if not ordered_clusters:
        raise ValueError("No valid clusters found to plot.")

    n_clusters = len(ordered_clusters)

    def block_weights(src_id, tgt_id):
        src_neurons, tgt_neurons = sets_int[src_id], sets_int[tgt_id]
        if src_neurons.size == 0 or tgt_neurons.size == 0:
            return np.array([])
        return W[np.ix_(src_neurons, tgt_neurons)].ravel()

    heatmap_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        _, src_id = ordered_clusters[i]
        for j in range(n_clusters):
            _, tgt_id = ordered_clusters[j]
            weights = block_weights(src_id, tgt_id)
            heatmap_matrix[i, j] = np.mean(weights) if weights.size > 0 else 0.0

    v_max = np.max(np.abs(heatmap_matrix))
    im = ax.imshow(
        heatmap_matrix,
        cmap="bwr",
        vmin=-v_max,
        vmax=v_max,
        aspect="auto"
    )

    # Cluster tick labels
    ax.set_xticks(np.arange(n_clusters))
    ax.set_yticks(np.arange(n_clusters))
    ax.set_xticklabels(
        [cid for _, cid in ordered_clusters],
        rotation=90,
        fontsize=s["heatmap_tick_fs"]
    )
    ax.set_yticklabels(
        [cid for _, cid in ordered_clusters],
        fontsize=s["heatmap_tick_fs"]
    )

    ax.set_title(title, fontsize=s["title_fs"], fontweight="bold", pad=s["title_pad"])
    ax.set_xlabel("Target cluster", fontsize=s["label_fs"])
    ax.set_ylabel("Source cluster", fontsize=s["label_fs"])
    ax.tick_params(axis="both", which="major", labelsize=s["heatmap_tick_fs"])

    # Find boundaries between contexts
    boundaries = [
        i for i in range(n_clusters - 1)
        if ordered_clusters[i][0] != ordered_clusters[i + 1][0]
    ]
    for b in boundaries:
        ax.axhline(b + 0.5, linewidth=1.5, color="black")
        ax.axvline(b + 0.5, linewidth=1.5, color="black")

    # ---- Compute context spans ----
    context_spans = []
    start = 0
    current_ctx = ordered_clusters[0][0]

    for i in range(1, n_clusters):
        ctx = ordered_clusters[i][0]
        if ctx != current_ctx:
            context_spans.append((current_ctx, start, i - 1))
            start = i
            current_ctx = ctx
    context_spans.append((current_ctx, start, n_clusters - 1))

    # ---- Add centered context labels on top ----
    bottom_ctx_ax = ax.secondary_xaxis("bottom")
    bottom_ctx_ax.set_xticks([(start + end) / 2 for _, start, end in context_spans[:-1]])
    bottom_ctx_ax.set_xticklabels(
        [context_to_context_name.get(ctx, ctx) for ctx, _, _ in context_spans[:-1]],
        fontsize=s["label_fs"]
    )
    bottom_ctx_ax.tick_params(axis="x", length=0, pad=50)
    
    # bottom_ctx_ax.spines["bottom"].set_position(("outward", 30))


    # move the secondary bottom axis further down
    
    # ---- Add centered context labels on the left ----
    # Put them as text to the left of the heatmap
    # for ctx, start, end in context_spans:
    #     center = (start + end) / 2
    #     ax.text(
    #         -1.2, center,
    #         context_to_context_name.get(ctx, ctx),
    #         ha="right",
    #         va="center",
    #         fontsize=s["label_fs"],
    #         fontweight="bold",
    #         transform=ax.transData,
    #         clip_on=False
    #     )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean weight", fontsize=s["cbar_label_fs"])
    cbar.ax.tick_params(labelsize=s["cbar_tick_fs"])

    return {
        "heatmap_matrix": heatmap_matrix,
        "ordered_clusters": ordered_clusters,
        "image": im
    }
    
def _add_panel_label(ax, label, x=-0.09, y=1.1, fontsize=20, fontweight="bold"):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight=fontweight,
    )

# -----------------------------
# Master 2x2 plot
# -----------------------------
def plot_network_summary_2x2(
    model,
    loader,
    activations=None,
    contexts=None,
    noiseless=True,
    device="cuda",
    threshold_ratio=0.5,
    variance_mode="var",
    latent_names=("Shape", "Colour", "Position"),
    style=None,
    
):
    s = _style(style)

    meta = compute_cluster_metadata(
        model=model,
        loader=loader,
        activations=activations,
        contexts=contexts,
    )

    cluster_to_neurons = meta["cluster_to_neurons"]
    context_to_cluster = meta["context_to_cluster"]
    sorted_order = meta["sorted_order"]

    fig, axs = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    # fig.set_constrained_layout_pads(hspace=0.25)
    axs = axs.ravel()
    rnn_cfg = copy.deepcopy(model.rnn_cfg)
    rnn_cfg.noise = 0.0
    noiseless_model =  model.create_new_instance(new_params={'rnn_cfg': rnn_cfg}).to(DEVICE)


    all_neurons_groups = {}
    for k in cluster_to_neurons:
        all_neurons_groups[str(k)] = cluster_to_neurons[k]

    # (3) active clusters over time
    plot_num_active_clusters_over_time_on_ax(
        dataloader=loader,
        model=noiseless_model,
        context_to_cluster=context_to_cluster,
        cluster_to_neurons=cluster_to_neurons,
        ax=axs[0],
        device=device,
        threshold_ratio=threshold_ratio,
        variance_mode=variance_mode,
        latent_names=latent_names,
        style=s,
        title="Number of active\n clusters for each latent",
    )
    _add_panel_label(axs[0], "A")

    plot_mean_cluster_activity_on_ax(
        model=noiseless_model,
        loader=loader,
        cluster_to_neurons=cluster_to_neurons,
        cluster_order=sorted_order,
        context_to_cluster=context_to_cluster,
        ax=axs[1],
        noiseless=noiseless,
        context_idx=0,
        device=device,
        style=s,
        title="Mean cluster activity",
    )
    _add_panel_label(axs[1], "B")

    plot_cosine_similarity_between_input_and_recurrent_input(
        model=noiseless_model,
        loader=loader,
        latent_idx=0,
        dynamic_idx=3,
        ax=axs[2],
        context_to_cluster=context_to_cluster,
        cluster_to_neurons=cluster_to_neurons,
    )
    _add_panel_label(axs[2], "C")

    # (4) context/cluster recurrent weight heatmap
    recurrent_weight = model.rnn.state_dict()["weight_hh.weight"].detach().cpu().numpy()
    all_neurons_groups = {str(k): v for k, v in cluster_to_neurons.items()}

    plot_context_weights_heatmaps_on_ax(
        W=recurrent_weight,
        all_neurons_groups=all_neurons_groups,
        context_to_cluster=context_to_cluster,
        ax=axs[3],
        style=s,
        title="Mean weights across clusters",
        fig=fig,
    )
    _add_panel_label(axs[3], "D")

    return fig, axs, meta