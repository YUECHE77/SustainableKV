import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class HybridKVCluster():

    def __init__(self, window_size=64, max_capacity_prompt=256 + 64, kernel_size=5, pooling='avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0

        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size=64, max_capacity_prompt=256 + 64, kernel_size=5, pooling='avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0

        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]

        bsz, num_heads, q_len, head_dim = query_states.shape

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            prefix_len = q_len - self.window_size

            q_win = query_states[..., -self.window_size:, :]  # [B, H, window_size, head_dim]
            k_hist = key_states[:, :, :-self.window_size, :]  # [B, H, prefix_len, head_dim] 
            attn_weights = torch.matmul(q_win, k_hist.transpose(2, 3)) / math.sqrt(head_dim)  # [B, H, window_size, prefix_len]

            if attention_mask is not None:
                mask_slice = attention_mask[..., -self.window_size:, :prefix_len]
                attn_weights += mask_slice

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights.sum(dim=-2)  # [B, H, prefix_len]
            prefix_value_states = value_states[:, :, :-self.window_size, :]  # [B, H, prefix_len, head_dim]
            v_norms = torch.linalg.norm(prefix_value_states + 1e-9, ord=2, dim=-1)  # [B, H, prefix_len]
            score_value_aware = attn_weights_sum * v_norms  # [B, ]

            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(score_value_aware, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(score_value_aware, kernel_size=self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            
            attn_cache_b1 = attn_cache[0]  # [H, prefix_len]

            # 1. 跨头聚合分数 (例如，取每个历史位置在所有头中的最大分数)
            aggregated_scores, _ = torch.max(attn_cache_b1, dim=0)  # [prefix_len, ]
            num_to_keep = self.max_capacity_prompt - self.window_size
            # 确保 k 不大于可用元素数量
            k_to_select = min(num_to_keep, aggregated_scores.numel())

            if k_to_select > 0:
            # 2. 从聚合后的分数中选出 Top-K 索引
                _, shared_indices = torch.topk(aggregated_scores, k_to_select, dim=-1)  # [k_to_select, ]
                indices = shared_indices.view(1, 1, k_to_select, 1).expand(1, num_heads, k_to_select, head_dim)
            else:
                indices = torch.empty((1, num_heads, 0, head_dim), dtype=torch.long, device=attn_cache.device)
                
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)  # [B, H, max_cache_size - window_size, head_dim]
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)  # [B, H, max_cache_size - window_size, head_dim]
            k_cur = key_states[:, :, -self.window_size:, :]  # [B, H, window_size, head_dim]
            v_cur = value_states[:, :, -self.window_size:, :]  # [B, H, window_size, head_dim]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)  # [B, H, max_cache_size, head_dim]
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)  # [B, H, max_cache_size, head_dim]

            return key_states, value_states

def init_hybridkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'

    self.kv_cluster = HybridKVCluster( 
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling
    )