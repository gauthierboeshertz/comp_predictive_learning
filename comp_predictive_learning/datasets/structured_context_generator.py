import random
from itertools import product

def _create_full_support(all_contexts, subsample_ratio):
    random.shuffle(all_contexts)
    train_size = int(len(all_contexts) * subsample_ratio)
    return all_contexts[:train_size]


def create_structured_contexts(config):
    all_context_names = list(config.dataset.contexts)
    not_used_context_names = []
    context_names = []
    for name in all_context_names:
        ctxt = config.dataset.get(name)
        if not len(ctxt):
            not_used_context_names.append(name)
        else:
            context_names.append(name)
    value_lists = [config.dataset.get(name) for name in context_names]
    all_contexts = [dict(zip(context_names, values)) for values in product(*value_lists)]
    
    if config.support_type == 'full' and config.subsample_contexts_ratio >= 1:
        print("Subsampling ratio is >= 1, using all contexts as training contexts.")
        return all_contexts, all_contexts
    
    train_contexts = []

    train_contexts = _create_full_support(all_contexts, config.subsample_contexts_ratio)

    train_set = {tuple(sorted(d.items())) for d in train_contexts}
    val_contexts = [ctx for ctx in all_contexts if tuple(sorted(ctx.items())) not in train_set]
    
    for train_context in train_contexts:
        for name in not_used_context_names:
            train_context[name] = 0
    for val_context in val_contexts:
        for name in not_used_context_names:
            val_context[name] = 0
        assert val_context not in train_contexts, "Validation context found in training contexts."
        
    print(f"--- Generating Support: '{config.support_type}' ---")
    print(f"Total contexts: {len(all_contexts)}")
    print(f"Train contexts: {len(train_contexts)}")
    print(f"Validation contexts: {len(val_contexts)}")
    
    return train_contexts, val_contexts
