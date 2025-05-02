import torch
import torch.nn.functional as F
import torch.nn as nn

import math

class SustainableKV:
    def __init__(self, 
                 window_size=32,
                 subseq_len=256,
                 attn_sink_tok=4,
                 desired_cache_size=2048,
                 pooling=None,
                 kernel_size=7,
                 recycling_percent=0.1,):
        
        self.window_size = window_size
        self.desired_cache_size = desired_cache_size
        assert self.window_size < self.desired_cache_size, (
            f'Window size ({self.window_size}) cannot exceed desired cache size ({self.desired_cache_size})'
        )

        self.subseq_len = subseq_len
        self.attn_sink_tok = attn_sink_tok

        if pooling is not None:
            assert pooling in ['avgpool', 'maxpool'], (
                f'only support avgpool and maxpool, but got {pooling}.'
            )
            self.kernel_size = kernel_size
            self.pooling = pooling
        
        self.recycling_percent = recycling_percent
    
    def reset(self, 
              window_size=32,
              subseq_len=256,
              attn_sink_tok=4,
              desired_cache_size=2048,
              pooling=None,
              kernel_size=7,
              recycling_percent=0.1,):
        
        self.window_size = window_size
        self.desired_cache_size = desired_cache_size
        assert self.window_size < self.desired_cache_size, (
            f'Window size ({self.window_size}) cannot exceed desired cache size ({self.desired_cache_size})'
        )

        self.subseq_len = subseq_len
        self.attn_sink_tok = attn_sink_tok

        if pooling is not None:
            assert pooling in ['avgpool', 'maxpool'], (
                f'only support avgpool and maxpool, but got {pooling}.'
            )
            self.kernel_size = kernel_size
            self.pooling = pooling
        
        self.recycling_percent = recycling_percent
    
    def allocate_budget(self, total_budget, num_subseq, all_subseq_len, prioritize_recent=True):
        """
        Allocate budget for each subsequence (the top-k for each subsequence).

        You can choose to either keep as many recent subseq as possible (similar to the idea of 
        sliding window, keep the most recent tokens) or evenly distribute the remaining to all subseq.
        """
        assert total_budget <= sum(all_subseq_len), (
            f"total_budget ({total_budget}) exceeds total capacity ({sum(all_subseq_len)})."
        )
        
        avg_budget = total_budget // num_subseq
        budgets = [min(avg_budget, l) for l in all_subseq_len]
        remaining = total_budget - sum(budgets)

        if prioritize_recent:
            for i in range(num_subseq-1, -1, -1):  # most recent subseq has priority
                can_add = all_subseq_len[i] - budgets[i]
                add = min(remaining, can_add)
                budgets[i] += add
                remaining -= add
                if remaining == 0:
                    break
        else:
            while remaining > 0:
                updated = False

                for i in range(num_subseq-1, -1, -1):
                    if budgets[i] < all_subseq_len[i]:
                        budgets[i] += 1
                        remaining -= 1
                        updated = True
                        if remaining == 0:
                            break

                if not updated:
                    break
        
        return budgets

    def recycle_cache(self, 
                      selected_indices: torch.Tensor, 
                      query_state: torch.Tensor, 
                      recycle_budget: int):
        """
        :param selected_indices: Relative position (indices) with shape [B, H, subseq_budget]
        :param query_state: Sliced query_state with shape [B, H, subseq_len, head_dim]
        :param recycle_budget: Number of tokens to be recycled
        """
        bsz, num_heads, subseq_len, head_dim = query_state.shape
        all_indices = torch.arange(subseq_len, dtype=torch.long, device=query_state.device)[None, None, :].expand(bsz, num_heads, -1)  # [B, H, subseq_len]
        
        all_indices_expand = all_indices.unsqueeze(-1)  # [B, H, subseq_len, 1]
        selected_indices_expand = selected_indices.unsqueeze(-2)  # [B, H, 1, subseq_budget]

        match = all_indices_expand == selected_indices_expand  # [B, H, subseq_len, subseq_budget]
        isin_mask = match.any(dim=-1)  # [B, H, subseq_len]
        evict_mask = ~isin_mask  # [B, H, subseq_len]

        drop_indices = all_indices.masked_select(evict_mask).view(bsz, num_heads, -1)  # deleted pool: [B, H, subseq_len - subseq_budget]

        drop_indices_expand = drop_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [B, H, subseq_len - subseq_budget, head_dim]
        selected_indices_expand = selected_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [B, H, subseq_budget, head_dim]
        query_selected = query_state.gather(dim=2, index=selected_indices_expand)  # [B, H, subseq_budget, head_dim]
        query_dropped = query_state.gather(dim=2, index=drop_indices_expand)  # [B, H, subseq_len - subseq_budget, head_dim]
        
        # [B, H, subseq_len - subseq_budget, subseq_budget] = [B, H, num_dropped_tok, num_selected_tok]
        eps = 1e-8
        query_selected_norm = torch.norm(query_selected, dim=-1, keepdim=True).clamp(min=eps)
        query_dropped_norm = torch.norm(query_dropped, dim=-1, keepdim=True).clamp(min=eps)
        cos_simi = (query_dropped / query_dropped_norm) @ (query_selected / query_selected_norm).transpose(-1, -2)
        avg_cos_simi = cos_simi.mean(dim=-1)  # [B, H, num_dropped_tok]

        most_similar_tok_idx = avg_cos_simi.topk(recycle_budget, dim=-1).indices  # [B, H, recycle_budget]
        recycle_indices = drop_indices.gather(dim=-1, index=most_similar_tok_idx)  # [B, H, recycle_budget]

        new_selected_indices = torch.concat([selected_indices, recycle_indices], dim=-1)  # [B, H, subseq_budget + recycle_budget]

        return new_selected_indices

    def update_kv(self,
                  query_states: torch.Tensor,
                  key_states: torch.Tensor,
                  value_states: torch.Tensor):
        """
        The core implementation of SustainableKV. Only applied on the prefilling stage.

        Step 1: KV cache eviction.
        Step 2: Evicted tokens recyclying.

        :param query_states: Shape [B, H, q_len, head_dim]
        :param key_states:   Shape [B, H, q_len, head_dim]
        :param value_states: Shape [B, H, q_len, head_dim]
        """
        assert key_states.shape[-2] == query_states.shape[-2]

        bsz, num_heads, q_len, head_dim = query_states.shape

        if q_len < self.desired_cache_size:
            # No need for eviction if the sequence length is already short
            return key_states, value_states
        else:
            query_obs_window = query_states[:, :, -self.window_size:, :]  # [B, H, win_size, head_dim]
            key_prefix = key_states[:, :, :-self.window_size, :]  # [B, H, prefix_size, head_dim]

            attn_weights = torch.matmul(query_obs_window, key_prefix.transpose(2, 3)) / math.sqrt(head_dim)  # [B, H, win_size, prefix_size]
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.device)
            attn_weights_sum = attn_weights.sum(dim=-2)  # [B, H, prefix_size]

            if hasattr(self, 'pooling') and hasattr(self, 'kernel_size'):
                if self.pooling == 'avgpool':
                    attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)  # [B, H, prefix_size]
                elif self.pooling == 'maxpool':
                    attn_cache = F.max_pool1d(attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)  # [B, H, prefix_size]
                else:
                    raise ValueError('Pooling method not supported')
            else:
                attn_cache = attn_weights_sum
            
            num_candidates = attn_cache.shape[-1] - self.attn_sink_tok  # number of all the candidates -> length of (q_len - attn_sink - window_size)
            num_subseq = math.ceil(num_candidates / self.subseq_len)
            last_seq_len = num_candidates - (num_subseq - 1) * self.subseq_len
            all_subseq_len = [self.subseq_len] * num_subseq
            all_subseq_len[-1] = last_seq_len
            assert sum(all_subseq_len) == num_candidates, (
                f'Summation of all of the subsequences ({sum(all_subseq_len)}) should equals to number of all the candidates ({num_candidates})'
            )

            total_budget = self.desired_cache_size - self.attn_sink_tok - self.window_size  # the total budget you can allocate < num_candidates
            all_subseq_budget = self.allocate_budget(total_budget, num_subseq, all_subseq_len, prioritize_recent=True)
            assert sum(all_subseq_budget) == total_budget

            all_subseq_select_budget = []
            all_subseq_recyele_budget = []
            for budget in all_subseq_budget:
                selected_tok_budget = int((1 - self.recycling_percent) * budget)
                all_subseq_select_budget.append(selected_tok_budget)  # number of selected tokens
                all_subseq_recyele_budget.append(budget - selected_tok_budget)  # number of recycle tokens

            candidates_shift = self.attn_sink_tok  # 4
            attn_sink_indices = torch.arange(self.attn_sink_tok, dtype=torch.long, device=query_states.device)[None, None, :].expand(bsz, num_heads, -1)
            top_k_indices = [attn_sink_indices, ]
            for i in range(num_subseq):
                subseq_len = all_subseq_len[i]
                subseq_select_budget = all_subseq_select_budget[i]
                subseq_recycle_budget = all_subseq_recyele_budget[i]

                # In case we only have one subseq (subseq_len > num_candidates) -> seems like no need for any extra operation ...
                candidates_start = i * self.subseq_len + candidates_shift
                selected_indices = attn_cache[:, :, candidates_start : candidates_start + subseq_len].topk(subseq_select_budget, dim=-1).indices  # [B, H, subseq_budget]

                final_indices = self.recycle_cache(
                    selected_indices=selected_indices,
                    query_state=query_states[:, :, candidates_start : candidates_start + subseq_len, :],
                    recycle_budget=subseq_recycle_budget
                )
                
                indices_shift = sum(all_subseq_len[:i]) + candidates_shift  # TODO: this should be the same as candidates_start
                final_indices += indices_shift  # the absolute indices
                top_k_indices.append(final_indices)
            
            top_k_indices_concat = torch.concat(top_k_indices, dim=-1).to(query_states.device)  # [B, H, total_budget + attn_sink]
            assert top_k_indices_concat.shape[-1] + self.window_size == self.desired_cache_size
            top_k_indices_selected = top_k_indices_concat.unsqueeze(-1).expand(-1, -1, -1, head_dim)  # [B, H, total_budget + attn_sink, head_dim]

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=top_k_indices_selected)  # [B, H, total_budget + attn_sink, head_dim]
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=top_k_indices_selected)  # [B, H, total_budget + attn_sink, head_dim]
            k_cur = key_states[:, :, -self.window_size:, :]  # [B, H, window_size, head_dim]
            v_cur = value_states[:, :, -self.window_size:, :]  # [B, H, window_size, head_dim]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)  # [B, H, desired_cache_size, head_dim]
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)  # [B, H, desired_cache_size, head_dim]

            return key_states, value_states

def init_sustainablekv(self):
    """
    Initialize SustainableKV

    :param self: transformers.models.llama.modeling_llama.LlamaFlashAttention2()
    """
    if not hasattr(self, 'cache_killer'):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'desired_cache_size'):
            self.config.desired_cache_size = 2048
        if not hasattr(self.config, 'subseq_len'):
            self.config.subseq_len = 1024
        if not hasattr(self.config, 'attn_sink_tok'):
            self.config.attn_sink_tok = 4
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'maxpool'
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 7
        if not hasattr(self.config, 'recycling_percent'):
            # 10% of the total selected tokens are recycled from the deleted pool
            self.config.recycling_percent = 0.5
    
    self.cache_killer = SustainableKV(
        window_size=self.config.window_size,
        subseq_len=self.config.subseq_len,
        attn_sink_tok=self.config.attn_sink_tok,
        desired_cache_size=self.config.desired_cache_size,
        pooling=self.config.pooling,
        kernel_size=self.config.kernel_size,
        recycling_percent=self.config.recycling_percent,
    )