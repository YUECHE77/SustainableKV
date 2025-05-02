
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
import math
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
# perform qk calculation and get indices
# this version will not update in inference mode

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class SnapKVCluster():
    def __init__(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def reset(self, window_size = 64, max_capacity_prompt = 256 + 64, kernel_size = 5, pooling = 'avgpool'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
              # —— debug before compression ——
        # 原始 prompt KV 大小
        orig_kv = key_states.shape[-2]
        print(f"[SnapKV] 原始 KV 大小 = {orig_kv}")
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                    # —— debug after compression ——
            new_kv = key_states.shape[-2]
            compression = 1.0 - new_kv / orig_kv
            print(f"[SnapKV] 压缩后 KV 大小 = {new_kv}，压缩率 = {compression*100:.2f}%")
            exit()
            return key_states, value_states

def init_snapkv(self):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
    self.kv_cluster = SnapKVCluster( 
        window_size = self.config.window_size, 
        max_capacity_prompt = self.config.max_capacity_prompt, 
        kernel_size = self.config.kernel_size,
        pooling = self.config.pooling
        )

def init_ValueAwareRandomCompressor(self):

    if hasattr(self, "kv_cluster"):
        return


    cfg = self.config



    window_size         = getattr(cfg, "window_size", 64)
    kernel_size         = getattr(cfg, "kernel_size", 7)
    pooling             = getattr(cfg, "pooling", "maxpool")
    max_capacity_prompt = getattr(cfg, "max_capacity_prompt", 4096)
    random_num          = getattr(cfg, "random_num", 64)
    value_norm_ord      = getattr(cfg, "value_norm_ord", 1)


    self.kv_cluster = ValueAwareRandomCompressor(
        layer_idx           = self.layer_idx,
        window_size         = window_size,
        kernel_size         = kernel_size,
        pooling             = pooling,
        max_capacity_prompt = max_capacity_prompt,
        random_num          = random_num,
        value_norm_ord      = value_norm_ord,
    )


    if self.layer_idx == 1: # 或者 self.layer_idx == 0
        print(f"[ValueAwareRandomCompressor CFG (Layer {self.layer_idx})]:")
        print(f"  window_size: {window_size}")
        print(f"  kernel_size: {kernel_size}")
        print(f"  pooling: {pooling}")
        print(f"  max_capacity_prompt: {max_capacity_prompt}")
        print(f"  random_num: {random_num}")
        print(f"  value_norm_ord: {value_norm_ord}")

class ValueAwareRandomCompressor:

    def __init__(self, layer_idx, max_capacity_prompt, window_size, kernel_size,
                 pooling='maxpool', random_num=64, value_norm_ord=1):
        self.layer_idx = layer_idx
        self.max_capacity_prompt = max_capacity_prompt
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.random_num = random_num
        self.value_norm_ord = value_norm_ord

        # 可选：用于记录统计数据
        self.use_time = 0
        self.orig_kvs = []
        self.new_kvs = []# 记录第一个 batch item 的 new_kv

    @torch.no_grad()
    def update_kv(self, key_states, query_states, value_states,
                  attention_mask=None, num_key_value_groups=None,
                  token_embeddings=None):

        bsz, num_heads, seq_len, head_dim = key_states.shape
        orig_kv = seq_len


        is_prompt_processing = key_states.shape[-2] == query_states.shape[-2]


        if self.max_capacity_prompt <= self.window_size:
            target_hist_len_config = 0
        else:
            target_hist_len_config = self.max_capacity_prompt - self.window_size


        if is_prompt_processing and seq_len > self.max_capacity_prompt and target_hist_len_config > 0:
            prefix_len = seq_len - self.window_size

            if prefix_len <= 0:
                return key_states, value_states

            # ---------- 1. Split History / Window ----------
            k_hist, k_win = key_states[:, :, :prefix_len], key_states[:, :, -self.window_size:]
            v_hist, v_win = value_states[:, :, :prefix_len], value_states[:, :, -self.window_size:]

            q_win = query_states[:, :, -self.window_size:]

            # ---------- 2. Calculate Forward Attention Score (attn_sum) ----------

            scores = torch.matmul(q_win, k_hist.transpose(2, 3)) / math.sqrt(head_dim)
            if attention_mask is not None:
                try:
                    # 提取对应的 mask 部分
                    mask_slice = attention_mask[..., -self.window_size:, :prefix_len]

                    if mask_slice.shape!= scores.shape:

                        mask_slice = mask_slice.expand_as(scores)
                    scores = scores + mask_slice
                except Exception as e:
                    # print(f"Warning L{self.layer_idx}: Failed to apply attention mask slice. Error: {e}")
                    pass
            attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)

            attn_sum = attn.sum(dim=-2) # (B, H, prefix_len)

            # ---------- 3. Calculate Value Norms ----------

            v_norms = torch.linalg.norm(v_hist + 1e-9, ord=self.value_norm_ord, dim=-1) # (B, H, prefix_len)

            # ---------- 4. Calculate Value-Aware Score ----------
            score_value_aware = attn_sum * v_norms # (B, H, prefix_len)

            # ---------- 5. Pooling ----------
            if self.kernel_size > 1 and prefix_len >= self.kernel_size:
                padding = self.kernel_size // 2
                if self.pooling == 'avgpool':
                    pooled = F.avg_pool1d(score_value_aware, kernel_size=self.kernel_size,
                                          padding=padding, stride=1)
                else: # Default maxpool
                    pooled = F.max_pool1d(score_value_aware, kernel_size=self.kernel_size,
                                          padding=padding, stride=1)

                if pooled.shape[-1]!= prefix_len:

                    pooled = pooled[..., :prefix_len]
            else:

                pooled = score_value_aware

            # ---------- 6. Select Indices (Value-Aware TopK + Random) - Per Batch Item ----------
            final_hist_indices_list = []
            actual_kept_hist_len_list =  []

            for b in range(bsz):

                current_target_hist_len = min(target_hist_len_config, prefix_len)


                num_random_target = min(self.random_num, current_target_hist_len)
                num_topk_target = current_target_hist_len - num_random_target
                num_topk_target = max(0, num_topk_target) # 确保非负


                unique_topk_indices = torch.empty(0, dtype=torch.long, device=key_states.device)
                if num_topk_target > 0 and prefix_len > 0:

                    scores_per_pos, _ = pooled[b].max(dim=0)


                    k_topk = min(num_topk_target, scores_per_pos.numel())
                    if k_topk > 0:
                        _, topk_indices = scores_per_pos.topk(k_topk, dim=-1)
                        unique_topk_indices = topk_indices


                current_unique_indices = unique_topk_indices
                num_needed = current_target_hist_len - current_unique_indices.numel()

                final_indices_sorted_b = torch.empty(0, dtype=torch.long, device=key_states.device)
                if num_needed <= 0:

                    final_indices_sorted_b, _ = torch.sort(current_unique_indices[:current_target_hist_len])
                else:

                    potential_indices = torch.arange(prefix_len, device=key_states.device)
                    mask = torch.ones(prefix_len, dtype=torch.bool, device=key_states.device)
                    if current_unique_indices.numel() > 0:
                        mask[current_unique_indices] = False
                    potential_indices = potential_indices[mask]

                    num_random_to_select = min(num_needed, potential_indices.numel())

                    # if num_random_to_select < num_needed and self.layer_idx == 1:
                    #     print(f"Warning L{self.layer_idx} B{b}: Not enough unique indices remaining for random selection. Needed {num_needed}, got {num_random_to_select}.")

                    if num_random_to_select > 0:
                        perm = torch.randperm(potential_indices.numel(), device=key_states.device)
                        chosen_random = potential_indices[perm[:num_random_to_select]]
                        combined = torch.cat([current_unique_indices, chosen_random])
                        final_indices_sorted_b, _ = torch.sort(combined)
                    else:
                        final_indices_sorted_b, _ = torch.sort(current_unique_indices)

                final_hist_indices_list.append(final_indices_sorted_b)
                actual_kept_hist_len_list.append(final_indices_sorted_b.numel())

            # ---------- 7. Gather Selected History KV (Corrected for Batch Processing) ----------
            k_sel_list = []
            v_sel_list = []

            max_kept_len = max(actual_kept_hist_len_list) if actual_kept_hist_len_list else 0



            for b in range(bsz):
                indices_b = final_hist_indices_list[b]
                K_hist_kept_b = indices_b.numel()

                if K_hist_kept_b > 0:
                    # Gather for item b using its specific indices
                    indices_to_gather_b = indices_b.view(1, 1, K_hist_kept_b, 1).expand(1, num_heads, -1, head_dim)
                    k_sel_b = k_hist[b:b+1].gather(2, indices_to_gather_b)
                    v_sel_b = v_hist[b:b+1].gather(2, indices_to_gather_b)


                    pad_len = max_kept_len - K_hist_kept_b
                    if pad_len > 0:
                        padding_shape_k = (1, num_heads, pad_len, head_dim)
                        padding_shape_v = (1, num_heads, pad_len, head_dim)
                        k_sel_b = torch.cat([k_sel_b, torch.zeros(padding_shape_k, dtype=k_sel_b.dtype, device=k_sel_b.device)], dim=2)
                        v_sel_b = torch.cat([v_sel_b, torch.zeros(padding_shape_v, dtype=v_sel_b.dtype, device=v_sel_b.device)], dim=2)

                else:
                    k_sel_b = torch.zeros((1, num_heads, max_kept_len, head_dim), dtype=key_states.dtype, device=key_states.device)
                    v_sel_b = torch.zeros((1, num_heads, max_kept_len, head_dim), dtype=value_states.dtype, device=value_states.device)

                k_sel_list.append(k_sel_b)
                v_sel_list.append(v_sel_b)


            if k_sel_list:
                k_sel = torch.cat(k_sel_list, dim=0) # Shape: (B, H, max_kept_len, D)
                v_sel = torch.cat(v_sel_list, dim=0) # Shape: (B, H, max_kept_len, D)
            else: # Handle case where bsz=0
                k_sel = torch.empty(bsz, num_heads, 0, head_dim, dtype=key_states.dtype, device=key_states.device)
                v_sel = torch.empty(bsz, num_heads, 0, head_dim, dtype=value_states.dtype, device=value_states.device)


            # ---------- 8. Concatenate Selected History + Window ----------
            # k_sel/v_sel now have shape (B, H, max_kept_len, D)
            # k_win/v_win have shape (B, H, window_size, D)
            key_states_out = torch.cat([k_sel, k_win], dim=2)   # Shape: (B, H, max_kept_len + window_size, D)
            value_states_out = torch.cat([v_sel, v_win], dim=2) # Shape: (B, H, max_kept_len + window_size, D)

            # ---------- 9. Final Calculations and User's Print (Adjusted for Batch) ----------

            new_kv = max_kept_len + self.window_size
            compression = 1.0 - new_kv / orig_kv if orig_kv > 0 else 0

            # Update statistics
            self.use_time += 1
            self.orig_kvs.append(orig_kv)
            self.new_kvs.append(new_kv) # Record the consistent new_kv size

            # Print stats (e.g., for layer 1)
            # if self.layer_idx == 1:
            #     print(f" 原始 KV 大小 = {orig_kv}")
            #     # Report target hist len based on config
            #     print(f" 目标历史长度 (配置) = {target_hist_len_config}")
            #     # Report average actual kept history length before padding
            #     avg_kept_hist_pre_pad = sum(actual_kept_hist_len_list) / bsz if bsz > 0 else 0
            #     print(f" 批次平均实际保留历史 (填充前) = {avg_kept_hist_pre_pad:.1f} (最大: {max_kept_len})")
            #     print(f" 压缩后 KV 大小 (填充后) = {new_kv} (目标 {self.max_capacity_prompt})，压缩率 = {compression*100:.2f}%")


            return key_states_out, value_states_out

        else:
            # 不需要压缩或不是 prompt 阶段
            return key_states, value_states    

    
    

def init_EnhancedKVCompressor(self):
    """在 LlamaAttention 实例上挂载 kv_cluster（若尚未挂载）"""
    if hasattr(self, "kv_cluster"):
        return


    cfg = self.config
    cfg.window_size         = 64
    cfg.chunk_size          = getattr(cfg, "chunk_size",           256)
    cfg.kernel_size         = getattr(cfg, "kernel_size",          7)
    cfg.pooling             = getattr(cfg, "pooling",              "maxpool")
    cfg.max_capacity_prompt = getattr(cfg, "max_capacity_prompt",  4096)


    cfg.static_num  = getattr(cfg, "static_num",  4)
    cfg.topk_num    = getattr(cfg, "topk_num",    56)
    cfg.random_num  = getattr(cfg, "random_num",  4)


    self.kv_cluster = EnhancedKVCompressor(
        layer_idx            = self.layer_idx,
        window_size          = cfg.window_size,
        chunk_size           = cfg.chunk_size,
        kernel_size          = cfg.kernel_size,
        pooling              = cfg.pooling,
        max_capacity_prompt  = cfg.max_capacity_prompt,
        static_num           = cfg.static_num,
        topk_num             = cfg.topk_num,
        random_num           = cfg.random_num,
    )
    if self.layer_idx == 1:
        print(f"CFG : {cfg}")



class EnhancedKVCompressor:

    def __init__(
        self,
        layer_idx: int,
        window_size: int = 64,
        chunk_size: int = 256,
        kernel_size: int = 7,
        pooling: str = "maxpool",   # "maxpool" | "avgpool"
        max_capacity_prompt: int = 4096,
        gamma: float = 0.2,
        *,

        static_num: int = 4,
        topk_num:   int = 4,
        random_num: int = 4,
    ):

        self.layer_idx           = layer_idx
        self.window_size         = window_size
        self.chunk_size          = chunk_size
        self.kernel_size         = kernel_size
        self.pooling             = pooling
        self.max_capacity_prompt = max_capacity_prompt
        self.gamma               = gamma


        self.static_num  = static_num
        self.topk_num    = topk_num
        self.random_num  = random_num
        self.log_once = True


    @torch.no_grad()
    def update_kv(self,
                  key_states,   # (B,H,T,D)
                  query_states,
                  value_states,
                  attention_mask,
                  num_key_value_groups):



        static_num  = getattr(self, "static_num", 4)
        topk_num    = getattr(self, "topk_num",   64)
        random_num  = getattr(self, "random_num", 4)
        chunk_size  = getattr(self, "chunk_size", 256)
        window_size = 64
        B, H, T, D  = query_states.shape

        if T < self.max_capacity_prompt:
            return key_states, value_states


        hist_len = T - window_size
        k_hist, k_win = key_states[:, :, :hist_len],        key_states[:, :, -window_size:]
        v_hist, v_win = value_states[:, :, :hist_len],      value_states[:, :, -window_size:]
        q_win          = query_states[:, :, -window_size:]


        attn = (q_win @ k_hist.transpose(2, 3)) / math.sqrt(D)        # (B,H,W,hist)

        mask = torch.full((window_size, window_size),
                          torch.finfo(attn.dtype).min,
                          device=attn.device)
        ar = torch.arange(window_size, device=attn.device)
        mask.masked_fill_(ar < (ar + 1).view(window_size, 1), 0)
        attn[:, :, :, -window_size:] += mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query_states.dtype)
        score_fwd = attn.sum(dim=-2)                                   # (B,H,hist)


        q_mean = q_win.mean(dim=-2)                                    # (B,H,D)
        score_rev = (k_hist * q_mean.unsqueeze(2)).sum(-1) / math.sqrt(D)  # (B,H,hist)


        score = score_fwd + score_rev                 # (B,H,hist)       


        gather_idx_all = []
        for ch_start in range(0, hist_len, chunk_size):
            ch_end   = min(ch_start + chunk_size, hist_len)
            L_chunk  = ch_end - ch_start

            static_idx = torch.arange(ch_start,
                                      ch_start + min(static_num, L_chunk),
                                      device=key_states.device)          # (S,)

            remain_mask = torch.ones(L_chunk, device=key_states.device, dtype=torch.bool)
            remain_mask[:static_idx.numel()] = False


            sc_chunk = score.mean(dim=1)[:, ch_start:ch_end]  # (B, chunk_len)


            sc_chunk = sc_chunk.masked_fill(~remain_mask, -1e4)
            k_top = min(topk_num, remain_mask.sum().item())
            top_idx_local = sc_chunk.topk(k_top, dim=-1).indices  # (B, k_top)
            top_idx = top_idx_local + ch_start  # 变全局 idx


            cand_mask = remain_mask.clone()
            cand_mask[top_idx_local] = False
            cand_idx_local = torch.nonzero(cand_mask, as_tuple=False).squeeze(1)
            r_num = min(random_num, cand_idx_local.numel())
            if r_num > 0:
                rand_perm = cand_idx_local[torch.randperm(cand_idx_local.numel(),
                                                          device=key_states.device)][:r_num]
                res_idx = rand_perm + ch_start                          # (r_num,)
            else:
                res_idx = torch.tensor([], device=key_states.device, dtype=torch.long)


            gather_idx_all.append(torch.cat([static_idx,
                                             top_idx[0],   # 只用 batch=0 拿顺序，再下面 broadcast
                                             res_idx]))


        gather_idx = torch.unique(torch.cat(gather_idx_all)).sort().values  # (K_total,)

        gather_idx_expand = gather_idx.view(1, 1, -1, 1).expand(B, H, -1, 1)


        k_sel = k_hist.gather(2, gather_idx_expand.expand(-1, -1, -1, D))
        v_sel = v_hist.gather(2, gather_idx_expand.expand(-1, -1, -1, D))


        key_states   = torch.cat([k_sel, k_win], dim=2)
        value_states = torch.cat([v_sel, v_win], dim=2)

        pad = self.max_capacity_prompt - key_states.size(2)
        if pad > 0:
            k_pad = key_states[:, :, -1:, :].expand(-1, -1, pad, -1)
            v_pad = value_states[:, :, -1:, :].expand_as(k_pad)
            key_states   = torch.cat([key_states, k_pad], dim=2)
            value_states = torch.cat([value_states, v_pad], dim=2)
            
        compressed_len = key_states.size(2)
        raw_len = T
        hist_token_kept = gather_idx.numel()
        window_token_kept = window_size
        pad_token_added = pad if pad > 0 else 0
        kept_token_count = hist_token_kept + window_token_kept
        compression_ratio = 1 - kept_token_count / raw_len
        # if self.layer_idx == 1:
        #     print(f"[enhance] 原始 KV 大小 = {orig_kv}")
        #     print(f"[enhance] 压缩后 KV 大小 = {new_kv}，压缩率 = {compression*100:.2f}%")
        return key_states, value_states

    
