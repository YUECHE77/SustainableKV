import torch
import torch.nn.functional as F
import torch.nn as nn

import math
from typing import Optional

class CacheKiller:
    def __init__(
        self,
        window_size: int = 32,
        desired_cache_size: int = 2048,
        value_aware: bool = False,
        pooling: Optional[str] = None,
        kernel_size: int = 7,
        recycling_rate: float = 0.1,
        recycle_headwise: bool = False,
        merge: Optional[str] = None,
    ):
        self.desired_cache_size = desired_cache_size
        self.window_size = window_size
        self.value_aware = value_aware
        self.recycling_rate = recycling_rate
        self.recycle_headwise = recycle_headwise

        if pooling is not None:
            assert pooling.lower() in ('avgpool', 'maxpool'), (
                f'[Pooling] Only support "avgpool" and "maxpool". But got {pooling.lower()}.'
            )
            self.kernel_size = kernel_size
            self.pooling = pooling.lower()
        
        if merge is not None:
            assert merge.lower() in ('pivot', 'average', 'weighted'), (
                f'[Merging] Only support "pivot", "average", and "weighted". But got {merge.lower()}.'
            )
            self.merge = merge.lower()
    
    @staticmethod
    def get_non_selected_indices(
        total_seq_len: int,
        selected_indices: torch.Tensor,
    ):
        """
        To get the indices for those deleted/evicted tokens.

        Args:
            total_seq_len (`int`): Total length of the sequence = num_selected_tok + num_evicted_tok
            selected_indices (`torch.Tensor`): Selected indices with shape [num_selected_tok, ] or [B, H, num_selected_tok]
        
        Returns:
            `torch.Tensor` Evicted token indices with shape [num_evicted_tok, ] or [B, H, num_evicted_tok]
        """
        assert selected_indices.dim() == 1 or selected_indices.dim() == 3, (
            'The shape of selected_indices should either be [num_selected_tok, ] or [B, H, num_selected_tok], '
            f'But got {selected_indices.shape}.'
        )

        all_indices = torch.arange(total_seq_len, dtype=torch.long, device=selected_indices.device)  # [total_seq_len, ]

        if selected_indices.dim() == 3:
            bsz, num_heads, _ = selected_indices.shape
            all_indices = all_indices[None, None, :].expand(bsz, num_heads, -1)  # [B, H, total_seq_len]

            all_indices_expand = all_indices.unsqueeze(-1)  # [B, H, total_seq_len, 1]
            selected_indices_expand = selected_indices.unsqueeze(-2)  # [B, H, 1, num_selected_tok]

            mask = all_indices_expand == selected_indices_expand  # [B, H, total_seq_len, num_selected_tok]
            isin_mask = mask.any(dim=-1)  # [B, H, total_seq_len]
            evict_mask = ~isin_mask  # [B, H, total_seq_len]

            non_selected_indices = all_indices.masked_select(evict_mask).view(bsz, num_heads, -1)  # [B, H, num_dropped_tok]
        else:
            all_indices_expand = all_indices.unsqueeze(-1)  # [total_seq_len, 1]
            selected_indices_expand = selected_indices.unsqueeze(0)  # [1, num_selected_tok]

            mask = all_indices_expand == selected_indices_expand  # [total_seq_len, num_selected_tok]
            isin_mask = mask.any(dim=-1)  # [total_seq_len, ]
            evict_mask = ~isin_mask  # [total_seq_len, ]

            non_selected_indices = all_indices.masked_select(evict_mask)  # [num_evicted_tok, ]

        return non_selected_indices
    
    def recycle_cache(
        self,
        selected_indices: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        recycle_budget: int,
    ):
        """
        This function contains the process of recycling and merging. We will use the key_state to 
        compute cosine similarity, and the function will eventually return compressed Key and Value.

        Args
            selected_indices (`torch.Tensor`): Selected indices with shape [num_selected_tok, ]
            key_states (`int`): Key matrix with shape [B, H, q_len, head_dim]
            value_states (`torch.Tensor`): Value matrix with shape [B, H, q_len, head_dim]
            recycle_budget (`torch.Tensor`): Number of tokens to be recycled

        Returns:
            `torch.Tensor` The compressed Key and Value matrix with shapes [B, H, q_len, desired_size]
        """
        bsz, num_heads, q_len, head_dim = key_states.shape

        # Potential edge case: recycle_budget = 0. Two possibilities: 
        # 1. Nothing to recycle (evicted pool is empty) -> implies nothing to merge; 
        # 2. I don't want to recycle -> No need for any additional actions
        if len(selected_indices) == q_len:
            assert recycle_budget == 0, (
                f'All the tokens in the subsequence (total {q_len}) are selected, '
                f'thus your recycle budget should be 0, but got {recycle_budget}.'
            )
            return key_states, value_states
        
        dropped_indices = self.get_non_selected_indices(q_len, selected_indices)  # [num_evicted_tok, ]

        key_dropped = key_states[:, :, dropped_indices, :]  # [B, H, num_evicted_tok, head_dim]
        key_selected = key_states[:, :, selected_indices, :]  # [B, H, num_selected_tok, head_dim]
        value_selected = value_states[:, :, selected_indices, :]  # [B, H, num_selected_tok, head_dim]

        # Compute Cosine Similarity
        key_selected_norm = torch.norm(key_selected, dim=-1, keepdim=True).clamp(min=1e-8)
        key_dropped_norm = torch.norm(key_dropped, dim=-1, keepdim=True).clamp(min=1e-8)
        cos_simi = (key_dropped / key_dropped_norm) @ (key_selected / key_selected_norm).transpose(-1, -2)  # [B, H, num_evicted_tok, num_selected_tok]

        max_cos_simi_values, max_cos_simi_indices = cos_simi.max(dim=-1)  # Both [B, H, num_evicted_tok]

        if self.recycle_headwise:
            most_similar_tok_idx = max_cos_simi_values.topk(recycle_budget, dim=-1, largest=True).indices  # [B, H, recycle_budget] -> Relative Indices
            recycle_indices = dropped_indices[most_similar_tok_idx].unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [B, H, recycle_budget, head_dim] -> Absolute Indices

            key_recycled = key_states.gather(dim=2, index=recycle_indices)  # [B, H, recycle_budget, head_dim]
            value_recycled = value_states.gather(dim=2, index=recycle_indices)  # [B, H, recycle_budget, head_dim]
        else:
            # TODO: Support batch inference
            max_cos_simi_values_b1 = max_cos_simi_values[0]
            max_value, _ = max_cos_simi_values_b1.max(dim=0)  # Both [num_evicted_tok, ]

            most_similar_tok_idx = max_value.topk(recycle_budget, dim=-1, largest=True).indices  # [recycle_budget, ] -> Relative Indices
            recycle_indices = dropped_indices[most_similar_tok_idx]  # [recycle_budget, ] -> Absolute Indices

            key_recycled = key_states[:, :, recycle_indices, :]  # [B, H, recycle_budget, head_dim]
            value_recycled = value_states[:, :, recycle_indices, :]  # [B, H, recycle_budget, head_dim]

        # ------------------------------ Merging ------------------------------
        able_to_merge = (recycle_budget + len(selected_indices) < q_len)

        if hasattr(self, 'merge') and able_to_merge:
            assert len(dropped_indices) == max_cos_simi_indices.shape[-1], (
                f'Inconsistant number of dropped tokens.'
                f'Got {len(dropped_indices)} and {max_cos_simi_indices.shape[-1]}'
            )

            num_dropped_tok = len(dropped_indices)

            # 1. Get the *SECOND* dropped tokens
            # Deleted pool: [B, H, num_dropped_tok - recycle_budget] = [B, H, sec_drop] / [sec_drop, ]
            sec_droppod_indices = self.get_non_selected_indices(num_dropped_tok, most_similar_tok_idx)
            sec_dropped_abs_indices = dropped_indices[sec_droppod_indices]  # [B, H, sec_drop] / [sec_drop, ]

            # TODO: Maybe check the threshold.
            if self.recycle_headwise:
                sec_dropped_abs_indices = sec_dropped_abs_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [B, H, sec_drop, head_dim]

                key_deleted = key_states.gather(dim=2, index=sec_dropped_abs_indices)  # [B, H, sec_drop, head_dim]
                value_deleted = value_states.gather(dim=2, index=sec_dropped_abs_indices)  # [B, H, sec_drop, head_dim]

                # 2. Get the indices of Source token (to merge from)
                max_indices = max_cos_simi_indices.gather(dim=-1, index=sec_droppod_indices)  # [B, H, sec_drop] -> relative indices
            else:
                key_deleted = key_states[:, :, sec_dropped_abs_indices, :]  # [B, H, sec_drop, head_dim]
                value_deleted = value_states[:, :, sec_dropped_abs_indices, :]  # [B, H, sec_drop, head_dim]

                # 2. Get the indices of Source token (to merge from)
                max_indices = max_cos_simi_indices[:, :, sec_droppod_indices]  # [B, H, sec_drop] -> relative indices
            
            # IMPORTANT: use max_indices finding src in "pivotal" merge, as well as conducting final merging.
            max_indices = max_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [B, H, sec_drop, head_dim]

            if self.merge == 'pivot':
                key_src = key_selected.gather(dim=2, index=max_indices)  # The source (Key) to merge: [B, H, sec_drop, head_dim]
                value_src = value_selected.gather(dim=2, index=max_indices)  # The source (Value) to merge  # [B, H, sec_drop, head_dim]

                # Conduct pivotal merge
                key_merge = (key_src + key_deleted) / 2  # [B, H, sec_drop, head_dim]
                value_merge = (value_src + value_deleted) / 2  # [B, H, sec_drop, head_dim]

            elif self.merge == 'average':
                # Conduct average merge
                key_merge = key_deleted  # [B, H, sec_drop, head_dim]
                value_merge = value_deleted  # [B, H, sec_drop, head_dim]

            else:
                # TODO: Maybe devided by std
                if self.recycle_headwise:
                    weights = max_cos_simi_values.gather(dim=-1, index=sec_droppod_indices).unsqueeze(-1)  # [B, H, sec_drop, 1]
                else:
                    weights = max_cos_simi_values[:, :, sec_droppod_indices].unsqueeze(-1)  # [B, H, sec_drop, 1]

                # Conduct weighted merge
                key_merge = key_deleted * weights
                value_merge = value_deleted * weights

            key_selected = key_selected.scatter_reduce(dim=2, index=max_indices, src=key_merge, reduce='mean', include_self=True)
            value_selected = value_selected.scatter_reduce(dim=2, index=max_indices, src=value_merge, reduce='mean', include_self=True)
        # ---------------------------------------------------------------------

        key_states_compressed = torch.concat([key_selected, key_recycled], dim=-2)
        value_states_compressed = torch.concat([value_selected, value_recycled], dim=-2)

        return key_states_compressed, value_states_compressed
    
    def update_kv(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor
    ):
        """
        The core implementation of the algorithm. Only applied in the prefilling stage.

        Step 1: KV cache eviction.\\
        Step 2: Evicted tokens recyclying and merging.

        Args:
            query_states (`torch.Tensor`): The query tensor with shape [B, H, q_len, head_dim]
            key_states (`torch.Tensor`): The key tensor with shape [B, H, q_len, head_dim]
            value_states (`torch.Tensor`): The value tensor with shape [B, H, q_len, head_dim]

        Returns:
            `tuple(torch.Tensor)` Compressed KV Cache with shape [B, H, desired_size, head_dim]
        """
        assert key_states.shape[-2] == query_states.shape[-2], 'Not in the prefilling stage.'

        bsz, num_heads, q_len, head_dim = query_states.shape

        if q_len < self.desired_cache_size:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)  # [B, H, window_size, q_len]
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)  # [B, H, window_size, q_len]
            attn_weights_sum = attn_weights.sum(dim=-2)  # [B, H, q_len]
            
            # 1. Value Aware
            # TODO: Try to normalize
            if self.value_aware:
                value_norm = torch.linalg.norm(value_states + 1e-8, ord=1, dim=-1)  # [B, H, q_len]
                attn_weights_sum *= value_norm  # [B, H, q_len]

            # 2. Pooling
            if hasattr(self, "pooling"):
                if self.pooling == 'avgpool':
                    attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)
                else:  # Max Pooling
                    attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                attn_cache = attn_weights_sum
            
            # 3. Non-Headwise Selection
            # TODO: Support batch inference
            attn_cache_b1 = attn_cache[0]  # [H, q_len]
            max_tok_scores, _ = attn_cache_b1.max(dim=0)  # [q_len, ]  TODO: Check mean()

            recycle_budget = int(self.recycling_rate * self.desired_cache_size)
            select_budget = self.desired_cache_size - recycle_budget
            selected_indices = max_tok_scores.topk(select_budget, dim=-1).indices  # [select_budget, ]

            # 4. Recycling and Merging
            key_states_compressed, value_states_compressed = self.recycle_cache(
                selected_indices=selected_indices,
                key_states=key_states,
                value_states=value_states,
                recycle_budget=recycle_budget
            )

            return key_states_compressed, value_states_compressed


def init_cachekiller(self):
    """
    Initialize CacheKiller
    
    Args:
        self (`transformers.models.llama.modeling_llama.LlamaFlashAttention2()`)
    """
    if not hasattr(self, 'cache_killer'):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'desired_cache_size'):
            self.config.desired_cache_size = 128
        if not hasattr(self.config, 'value_aware'):
            self.config.value_aware = False
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'recycling_rate'):
            self.config.recycling_rate = 0.0
        if not hasattr(self.config, 'recycle_headwise'):
            self.config.recycle_headwise = False
        if not hasattr(self.config, 'merge'):
            self.config.merge = None  # pivot, average, weighted, or None
    
    self.cache_killer = CacheKiller(
        window_size=self.config.window_size,
        desired_cache_size=self.config.desired_cache_size,
        value_aware=self.config.value_aware,
        pooling=self.config.pooling,
        kernel_size=self.config.kernel_size,
        recycling_rate=self.config.recycling_rate,
        recycle_headwise=self.config.recycle_headwise,
        merge=self.config.merge,
    )
