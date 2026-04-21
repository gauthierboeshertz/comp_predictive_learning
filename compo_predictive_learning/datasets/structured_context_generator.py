import random
from itertools import product

def _create_full_support(all_contexts, subsample_ratio):
    random.shuffle(all_contexts)
    train_size = int(len(all_contexts) * subsample_ratio)
    return all_contexts[:train_size]


def _train_contexts_cover_all_parts(train_contexts, config):
    """
    Check that for every active context variable, every possible value
    appears at least once in the training contexts.
    """
    all_context_names = list(config.dataset.contexts)

    for name in all_context_names:
        possible_values = config.dataset.get(name)

        if not len(possible_values):
            continue

        seen_values = {ctx[name] for ctx in train_contexts}
        for value in possible_values:
            if value not in seen_values:
                print(f"Value '{value}' of context '{name}' not covered in training contexts.")
                return False
    return True

def create_structured_contexts(config):
    def make_contexts():
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
            
        return train_contexts, val_contexts
    
    num_tried = 0
    while num_tried < 50:
        train_contexts, val_contexts = make_contexts()
        if _train_contexts_cover_all_parts(train_contexts, config):
            print(f"Found a good split after {num_tried+1} tries.")
            return train_contexts, val_contexts
        num_tried += 1
    raise ValueError("Could not find a good split of contexts after 50 tries. Consider increasing the subsample_contexts_ratio or checking the context definitions.")    
