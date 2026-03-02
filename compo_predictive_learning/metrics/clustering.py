import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn import metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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

# ── Data collection ────────────────────────────────────────────────────────────


# ── Clustering ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_optimal_n_clusters(model, loader=None, activations=None, contexts=None,
                           time_variance=True, max_clusters=25,device=None):
    """
    Find the best KMeans cluster count via silhouette score.
    Returns (best_k, scores, norm_variance, active_units, labels).
    """
    if activations is None:
        activations, contexts = get_rnn_activities_and_sources_for_loader_for_clustering(model, loader)

    # Per-context mean variance profile for each neuron
    _, inv = contexts.unique(dim=0, return_inverse=True)
    per_ctx = torch.stack([activations[inv == i] for i in range(inv.max() + 1)])
    per_ctx = per_ctx.permute(3, 0, 1, 2)  # (neurons, contexts, trials, time)

    dim = 3 if time_variance else 2
    var_profile = per_ctx.var(dim=dim).mean(dim=2)  # (neurons, contexts)

    active = var_profile.sum(1) > 0.001
    var_profile = var_profile[active]
    norm_var = (var_profile.T / (var_profile.max(1)[0] + 1e-6)).T

    X = norm_var.numpy()
    n_samples = X.shape[0]
    if n_samples < 2:
        print("Not enough samples for clustering.")
        return 1, None, None, None, None

    ks = list(range(2, min(max_clusters, n_samples)))
    scores = [
        metrics.silhouette_score(X, KMeans(k, algorithm="lloyd", random_state=0).fit_predict(X))
        for k in ks
    ]

    best_k = ks[np.argmax(scores)]
    labels = KMeans(best_k, random_state=0).fit_predict(X)
    return best_k, np.array(scores), norm_var, active, labels


# ── Loss utilities ─────────────────────────────────────────────────────────────

def get_loss_per_context(model, loader):
    """Return {context_tuple: mean_loss} for each context in the loader."""
    model.eval()
    model.loss_fn = model.new_loss_fn(reduction="none")
    loss_per_context = defaultdict(list)

    with torch.no_grad():
        for prim_in, _, contexts in loader:
            losses = model.loss(prim_in)[0].mean(dim=(0, 2, 3, 4))
            for idx, ctx in enumerate(contexts):
                loss_per_context[tuple(ctx.cpu().tolist())].append(losses[idx].item())

    return {ctx: np.mean(vals) for ctx, vals in loss_per_context.items()}


def compute_lesioned_cluster_losses(model, loader, active_units, labels, n_clusters):
    """Lesion each cluster in turn; return (cluster_losses, original_losses, cluster_to_neurons)."""
    active_idx = np.where(active_units.cpu().numpy())[0]
    cluster_to_neurons = {c: active_idx[labels == c] for c in range(n_clusters)}

    original_losses = get_loss_per_context(model.create_new_instance().to(DEVICE), loader)

    cluster_losses = {}
    for cid, neurons in cluster_to_neurons.items():
        lesioned = model.create_lesioned_instance(neurons).to(DEVICE)
        cluster_losses[cid] = get_loss_per_context(lesioned, loader)
        orig = np.mean(list(original_losses.values()))
        new  = np.mean(list(cluster_losses[cid].values()))
        print(f"Cluster {cid}: orig={orig:.4f}  lesioned={new:.4f}")

    return cluster_losses, original_losses, cluster_to_neurons


# ── Cluster analysis ───────────────────────────────────────────────────────────

def analyze_and_sort_clusters(norm_var, contexts_unique, labels,
                               selectivity_threshold=0.5, purity_threshold=0.90):
    """
    Classify clusters as selective to a context dimension or non-selective,
    then return a sorted ordering for plotting.
    """
    def to_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    norm_var        = to_numpy(norm_var)
    contexts_unique = to_numpy(contexts_unique)
    labels          = to_numpy(labels).reshape(-1)

    n_clusters  = labels.max() + 1
    n_contexts  = norm_var.shape[1]
    eps         = 1e-9

    # Mean activity profile per cluster
    profiles = np.array([
        norm_var[labels == c].mean(axis=0) if (labels == c).any() else np.zeros(n_contexts)
        for c in range(n_clusters)
    ])

    # Selectivity score
    peak       = profiles.max(axis=1)
    mean_other = (profiles.sum(axis=1) - peak) / max(n_contexts - 1, 1)
    selectivity = (peak - mean_other) / (peak + mean_other + eps)

    informative = np.where(selectivity > selectivity_threshold)[0]
    non_selective = np.where(selectivity <= selectivity_threshold)[0]

    # Peak-context map
    peak_map = {}
    if informative.size:
        for cid, ctx_i in zip(informative, np.argmax(profiles[informative], axis=1)):
            peak_map.setdefault(tuple(contexts_unique[ctx_i]), []).append(int(cid))

    # Variance explained by a single context dimension
    def var_explained(profile, dim_vals):
        total = np.var(profile)
        if total < eps:
            return 0.0
        resid = np.mean([np.var(profile[dim_vals == v]) for v in np.unique(dim_vals)])
        return 1.0 - resid / (total + eps)

    ctx_order = np.lexsort(tuple(contexts_unique[:, i] for i in range(contexts_unique.shape[1] - 1, -1, -1)))
    y_pos = np.empty_like(ctx_order); y_pos[ctx_order] = np.arange(len(ctx_order))

    n_dims = contexts_unique.shape[1]
    groups = {d: [] for d in range(n_dims)}
    mixed  = []

    for cid in informative:
        for d in range(n_dims):
            if var_explained(profiles[cid], contexts_unique[:, d]) > purity_threshold:
                groups[d].append(int(cid)); break
        else:
            mixed.append(int(cid))

    sorted_informative, group_boundaries, group_labels = [], [], []
    for d in range(n_dims):
        cids = groups[d]
        if not cids:
            continue
        dim_vals  = contexts_unique[:, d]
        uniq_vals = np.unique(dim_vals)
        key1 = [uniq_vals[np.argmax([profiles[c][dim_vals == v].sum() for v in uniq_vals])] for c in cids]
        key2 = [y_pos[int(np.argmax(profiles[c]))] for c in cids]
        order = np.lexsort((np.array(key2), np.array(key1)))
        sorted_informative.extend([cids[i] for i in order])
        group_boundaries.append(len(sorted_informative))
        group_labels.append(f"C{d}")

    non_sel = np.array(sorted(list(non_selective) + mixed), dtype=int)
    final_order = list(sorted_informative) + list(non_sel)
    if non_sel.size:
        group_boundaries.append(len(final_order))
        group_labels.append("Non-Selective")

    return final_order, group_labels, group_boundaries, peak_map, informative, profiles


# ── Plotting ───────────────────────────────────────────────────────────────────

def _to_numpy(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def plot_sorted_clusters(norm_var, contexts_unique, labels,
                          final_cluster_order, group_labels, group_boundaries,
                          fig, ax, put_y_label=True, group_name_map=None):
    norm_var        = _to_numpy(norm_var)
    contexts_unique = _to_numpy(contexts_unique)
    labels          = _to_numpy(labels).flatten()

    # Build sorted neuron / context indices
    neuron_order = np.concatenate([np.where(labels == c)[0] for c in final_cluster_order])
    ctx_order    = np.lexsort(tuple(contexts_unique[:, i] for i in range(contexts_unique.shape[1] - 1, -1, -1)))
    heat         = norm_var[neuron_order, :].T[ctx_order, :]

    im = ax.imshow(heat, aspect="auto", origin="upper", cmap="viridis")

    neurons_per_cluster = {c: (labels == c).sum() for c in np.unique(labels)}

    # Group neuron boundaries & dividing lines
    counts = [sum(neurons_per_cluster.get(c, 0) for c in final_cluster_order[s:e])
              for s, e in zip([0] + list(group_boundaries[:-1]), group_boundaries)]
    neuron_bounds = np.cumsum(counts)

    for b in neuron_bounds[:-1]:
        ax.axvline(b - 0.5, color="cyan", linewidth=2.5)

    line_colors = ["white", "yellow", "magenta", "orange", "red", "lime"]
    x_start = -0.5
    for i, (label, x_end) in enumerate(zip(group_labels, neuron_bounds - 0.5)):
        if not label.startswith("C"):
            x_start = x_end; continue
        d       = int(label[1:])
        dim_ctx = contexts_unique[ctx_order, d]
        diffs   = np.diff(dim_ctx)
        normal  = np.where(diffs != 0)[0][diffs[np.where(diffs != 0)[0]] > 0] + 0.5
        cycles  = np.where(diffs < 0)[0] + 0.5
        ax.hlines(np.setdiff1d(np.where(diffs != 0)[0] + 0.5, cycles),
                  x_start, x_end, colors=line_colors[i % len(line_colors)],
                  linestyles="--", linewidth=1.5, alpha=0.9)
        ax.hlines(cycles, x_start, x_end, colors=line_colors[i % len(line_colors)],
                  linestyles="-", linewidth=1.5, alpha=0.9)
        x_start = x_end

    # X ticks: cluster labels centred on each cluster block
    cluster_sizes  = [neurons_per_cluster.get(c, 0) for c in final_cluster_order]
    cluster_bounds = np.cumsum(cluster_sizes)
    centers        = cluster_bounds - np.array(cluster_sizes) / 2
    ax.set_xticks(centers); ax.set_xticklabels([str(c) for c in final_cluster_order], fontsize=16)
    ax.tick_params(axis="x", which="major", length=0)
    ax.set_xticks(cluster_bounds[:-1] - 0.5, minor=True)
    ax.tick_params(axis="x", which="minor", length=6)

    # Y ticks
    varying = [i for i in range(contexts_unique.shape[1]) if len(np.unique(contexts_unique[:, i])) > 1] \
              or list(range(contexts_unique.shape[1]))
    if put_y_label:
        ax.set_yticks(np.arange(len(contexts_unique)))
        ax.set_yticklabels([f"{[int(contexts_unique[ctx_order[r], i]) for i in varying]}"
                            for r in range(len(ctx_order))], fontsize=8)
        ax.set_ylabel("Contexts", fontsize=26)
    else:
        ax.set_yticks([]); ax.set_yticklabels([])

    # Group labels below x-axis
    group_centers = np.array([0] + list(neuron_bounds[:-1])) + np.array(counts) / 2
    for xc, lbl in zip(group_centers, group_labels):
        display = (group_name_map or {}).get(lbl, lbl).replace("Non-Selective", "NS")
        ax.annotate(display, xy=(xc, -0.04), xycoords=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=20)

    ax.set_xlabel("Neuron Clusters", fontsize=26, labelpad=30)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized variance", fontsize=24)
    fig.subplots_adjust(bottom=0.22)
    fig.set_tight_layout(True)
    return fig, ax


def plot_lesion_delta(original_losses, cluster_losses, fig, ax,
                      cluster_order=None, put_y_label=True):
    """Heatmap of (lesioned − original) loss per context × cluster."""
    contexts = sorted(original_losses)
    clusters = [c for c in (cluster_order or sorted(cluster_losses)) if c in cluster_losses]

    diff = np.array([[cluster_losses[cl][ctx] - original_losses[ctx]
                      for cl in clusters] for ctx in contexts])

    im = ax.imshow(diff, aspect="auto", origin="upper", cmap="binary")
    ax.set_xticks(range(len(clusters))); ax.set_xticklabels([str(c) for c in clusters], fontsize=16)
    ax.set_xlabel("Lesioned cluster", fontsize=26)

    ctx_arr    = np.array(contexts)
    varying    = [i for i in range(ctx_arr.shape[1]) if len(np.unique(ctx_arr[:, i])) > 1]
    ctx_labels = [f"{[int(ctx[i]) for i in varying]}" for ctx in contexts]

    if put_y_label:
        ax.set_yticks(range(len(contexts))); ax.set_yticklabels(ctx_labels, fontsize=8)
        ax.set_ylabel("Contexts", fontsize=26)
    else:
        ax.set_yticks([]); ax.set_yticklabels([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Lesioned Loss − Original Loss", fontsize=24)
    fig.set_tight_layout(True)
    return fig, ax


def plot_lesion_reconstructions(model, loader, cluster_to_neurons,
                                 t_start=1, t_end=4, num_samples=4):
    """Visual comparison of original vs. cluster-lesioned reconstructions."""
    inputs, _, contexts = next(iter(loader))
    inputs, contexts = inputs.to(DEVICE), contexts.to(DEVICE)

    def hide_ticks(ax_):
        ax_.set_xticks([]); ax_.set_yticks([])
        for s in ax_.spines.values(): s.set_visible(False)

    figs, axs = [], []
    for b in range(num_samples):
        sample = inputs[:, b:b+1]
        with torch.no_grad():
            baseline = model(sample)[0].detach()

        n_clusters = len(cluster_to_neurons)
        fig, ax = plt.subplots(2 + n_clusters, t_end - t_start,
                               figsize=((t_end - t_start) * 3, (2 + n_clusters) * 3),
                               constrained_layout=False)
        ax = np.atleast_2d(ax)
        fig.subplots_adjust(left=0.18, top=0.97, hspace=0.1, wspace=0.1)

        ax[0, 0].set_ylabel("Ground Truth",           fontsize=12, weight="bold", labelpad=12)
        ax[1, 0].set_ylabel("Original Reconstruction", fontsize=12, weight="bold", labelpad=12)

        for t_idx, t in enumerate(range(t_start, t_end)):
            ax[0, t_idx].imshow(sample[t + 1, 0].cpu().permute(1, 2, 0).numpy())
            ax[1, t_idx].imshow(baseline[t, 0].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
            hide_ticks(ax[0, t_idx]); hide_ticks(ax[1, t_idx])
            ax[-1, t_idx].set_xlabel(f"t={t+1}", fontsize=12)

        for i, cid in enumerate(sorted(cluster_to_neurons)):
            row = i + 2
            ax[row, 0].set_ylabel(f"Lesion C{cid}", fontsize=12, weight="bold", labelpad=12)
            lesioned = model.create_lesioned_instance(cluster_to_neurons[cid]).to(DEVICE)
            with torch.no_grad():
                recon = lesioned(sample)[0].detach()
            for t_idx, t in enumerate(range(t_start, t_end)):
                ax[row, t_idx].imshow(recon[t, 0].cpu().permute(1, 2, 0).clamp(0, 1).numpy())
                hide_ticks(ax[row, t_idx])

        figs.append(fig); axs.append(ax)

    return figs, axs, [int(c) for c in contexts[0][:3]]


# ── Top-level orchestration ────────────────────────────────────────────────────

def _dataset_group_name_map(config):
    if "sketch" in config.dataset.name:
        return {"C0": "shape", "C1": "color", "C2": "position"}
    if "ddd" in config.dataset.name:
        return {"C0": "shape", "C1": "floor hue", "C2": "wall hue", "C3": "scale"}
    return {}


def plot_cluster_and_lesions(config, model, loader):
    activations, contexts = get_rnn_activities_and_sources_for_loader_for_clustering(model, loader)
    best_k, scores, norm_var, active_units, labels = get_optimal_n_clusters(
        model, activations=activations, contexts=contexts, time_variance=False)
    unique_ctx, _ = contexts.unique(dim=0, return_inverse=True)

    if best_k == 1:
        print("Only one cluster found, skipping.")
        return None, None, None, None, None, None, None

    print(f"Optimal clusters: {best_k}, silhouette scores: {scores}")

    sorted_order, group_labels, group_boundaries, peak_map, informative, profiles = \
        analyze_and_sort_clusters(norm_var.cpu().numpy(), unique_ctx.cpu(), labels,
                                   selectivity_threshold=0.2, purity_threshold=0.5)

    is_auto = "auto" in config.model.type
    group_map = _dataset_group_name_map(config)

    fig_c, ax_c = plt.subplots(figsize=(10, 8))
    plot_sorted_clusters(norm_var.cpu().numpy(), unique_ctx.cpu(), labels,
                          sorted_order, group_labels, group_boundaries,
                          fig_c, ax_c, put_y_label=not is_auto, group_name_map=group_map)

    cluster_losses, orig_losses, cluster_to_neurons = compute_lesioned_cluster_losses(
        model, loader, active_units, labels, best_k)

    fig_l, ax_l = plt.subplots(figsize=(10, 8))
    plot_lesion_delta(orig_losses, cluster_losses, fig_l, ax_l,
                      cluster_order=sorted_order, put_y_label=not is_auto)

    figs, axs, ctx_sample = plot_lesion_reconstructions(
        model, loader, cluster_to_neurons, t_start=1, t_end=3, num_samples=1)

    return fig_c, ax_c, fig_l, ax_l, figs, axs, ctx_sample