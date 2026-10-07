"""Decode-time key/value zeroing used by this repository's DyCoke adaptation.

Zeroed keys remain in the attention softmax and cache allocation. This is cache
suppression, not token deletion and not the official DyCoke cache algorithm.
"""

from __future__ import annotations

import torch.nn.functional as F


class PruningTracker:
    """Track attention changes and zero low-attention visual cache positions."""

    def __init__(self, vision_token_mask=None, prune_ratio=0.2, sim_threshold=0.9, start_layer=3):
        if not 0 <= prune_ratio <= 1 or not -1 <= sim_threshold <= 1 or start_layer < 0:
            raise ValueError("Invalid cache suppression parameters")
        self.vision_token_mask = vision_token_mask
        self.prune_ratio = prune_ratio
        self.sim_threshold = sim_threshold
        self.start_layer = start_layer
        self._prev_attn = {}
        self._pruned = {}
        self.original_configs = []

    def record_attention_and_prune(self, layer_idx, attn_weights, cache):
        mask = self.vision_token_mask
        if (
            layer_idx < self.start_layer
            or mask is None
            or self.prune_ratio == 0
            or not mask.any()
            or attn_weights.shape[-2] != 1
        ):
            return
        if attn_weights.shape[0] != 1:
            raise ValueError("DyCoke adaptation supports batch size one")
        attention = attn_weights.mean(dim=1)[0, 0, : len(mask)]
        current = attention[mask.to(attention.device)].float()
        previous = self._prev_attn.get(layer_idx)
        if previous is not None and previous.shape == current.shape:
            similarity = F.cosine_similarity(previous[None], current[None]).item()
            if similarity < self.sim_threshold:
                n_prune = max(1, int(self.prune_ratio * len(current)))
                positions = mask.nonzero(as_tuple=True)[0].to(current.device)
                selected = positions[current.topk(n_prune, largest=False).indices]
                if hasattr(cache, "layers") and layer_idx < len(cache.layers):
                    keys = cache.layers[layer_idx].keys
                    values = cache.layers[layer_idx].values
                elif hasattr(cache, "key_cache") and layer_idx < len(cache.key_cache):
                    keys, values = cache.key_cache[layer_idx], cache.value_cache[layer_idx]
                else:
                    raise TypeError("Unsupported transformer cache layout for DyCoke suppression")
                if keys is None or values is None:
                    raise ValueError("Cannot suppress an uninitialized cache")
                keys[:, :, selected.to(keys.device), :] = 0
                values[:, :, selected.to(values.device), :] = 0
                self._pruned.setdefault(layer_idx, set()).update(selected.tolist())
        self._prev_attn[layer_idx] = current.detach().clone()

    @property
    def total_pruned(self):
        return sum(len(positions) for positions in self._pruned.values())


def install_dycoke_hooks(
    model,
    vision_token_mask=None,
    *,
    image_token_id=None,
    prune_ratio=0.2,
    sim_threshold=0.9,
    start_layer=3,
):
    """Install hooks on decoder self-attention; fail for unsupported model layouts.

    A model pre-hook obtains the vision mask from the initial input IDs. Attention
    modules, rather than decoder outputs, expose weights in Transformers 4.57.
    Eager attention is used only on single-token decode calls to avoid allocating
    a quadratic prefill attention matrix. Remove hooks in a ``finally`` block.
    """
    tracker = PruningTracker(vision_token_mask, prune_ratio, sim_threshold, start_layer)
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith("self_attn") and hasattr(module, "layer_idx")
    ]
    modules = [(name, module) for name, module in modules if module.layer_idx >= start_layer]
    if not modules:
        raise ValueError("No supported decoder self-attention modules at or after start_layer")
    if vision_token_mask is None and image_token_id is None:
        raise ValueError("image_token_id or vision_token_mask is required")
    handles = []
    try:
        configs_seen = set()
        for _, module in modules:
            config = module.config
            if id(config) not in configs_seen:
                tracker.original_configs.append((config, config._attn_implementation))
                configs_seen.add(id(config))

        if vision_token_mask is None:

            def capture_inputs(module, args, kwargs):
                ids = kwargs.get("input_ids", args[0] if args else None)
                if tracker.vision_token_mask is None and ids is not None:
                    if ids.shape[0] != 1:
                        raise ValueError("DyCoke adaptation supports batch size one")
                    tracker.vision_token_mask = (ids[0] == image_token_id).detach()

            handles.append(model.register_forward_pre_hook(capture_inputs, with_kwargs=True))

        def prepare_attention(module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is not None and hidden.shape[1] == 1:
                module.config._attn_implementation = "eager"

        def capture_attention(module, args, kwargs, output):
            for config, original in tracker.original_configs:
                if config is module.config:
                    config._attn_implementation = original
            if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                return
            cache = kwargs.get("past_key_values")
            if cache is None:
                cache = kwargs.get("past_key_value")
            if cache is not None:
                tracker.record_attention_and_prune(module.layer_idx, output[1], cache)

        for _, module in modules:
            handles.append(module.register_forward_pre_hook(prepare_attention, with_kwargs=True))
            handles.append(module.register_forward_hook(capture_attention, with_kwargs=True))
    except Exception:
        remove_hooks(handles, tracker)
        raise
    return tracker, handles


def remove_hooks(handles, tracker=None):
    """Remove handles and restore every attention configuration, including on errors."""
    for handle in handles:
        handle.remove()
    handles.clear()
    if tracker is not None:
        for config, implementation in tracker.original_configs:
            config._attn_implementation = implementation
        tracker.original_configs.clear()
