import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from assembly.ca_module import assembly, bipartite_soft_matching_x_w, do_nothing


def bipartite_soft_matching_x_w_scores(x, w):
    # We can only reduce by a maximum of 50% channels
    # metric shape: [B * N, C]
    b = x.shape[0]
    t = x.shape[1]

    with torch.no_grad():
        # xa, xb shape: [B * N, C/2]
        xa, xb = x[..., ::2], x[..., 1::2]
        # wa, wb shape: [cout, C/2]
        wa, wb = w[..., ::2], w[..., 1::2]
        xa_c, xb_c = xa.shape[1], xb.shape[1]

        # shape: [C/2, C/2]
        # fast version
        # xdist = (xa.t().reshape(xa_c, b, 1) - xb.reshape(1, b, xb_c)).sum(1)
        # score_ij = wi (yi - yj) / 2 + wj (yj - yi) / 2
        # score_a[i, :] = w:i xdist:

        xdist = torch.cdist(xa.t(), xb.t(), p=2.0)
        scores_a = torch.zeros(xa_c, xb_c, device=x.device)
        scores_b = torch.zeros(xa_c, xb_c, device=x.device)
        scores_fast = torch.zeros(xa_c, xb_c, device=x.device)
        for i in range(xb_c):
            scores_a[i, :] = (wa[:, i].unsqueeze(1) * xdist[i]).sum(0)
        for j in range(xb_c):
            scores_b[:, j] = (wb[:, j].unsqueeze(1) * (xdist[:, j])).sum(0)
        scores_fast = (scores_a + scores_b).pow(2)
        scores = scores_fast
    return scores


class CRModule(nn.Module):
    def __init__(self, layer, channel_ratio, act_fn=None):
        super().__init__()
        self.channel_ratio = channel_ratio
        self.outlier_channel_idx = None
        self.num_channels = 0
        self.num_additional_channels = 0

        # disassembly
        self.num_disassembly = None
        self.scaling_factors = None

        # assembly
        self.have_assembled = False
        self.src_idx = None
        self.dst_idx = None

        self.layer = layer
        self.act_fn = act_fn

    def find_outlier_channels(self, x_max, th):
        num_disassembly = torch.ceil(x_max / th)
        num_disassembly = torch.clamp(num_disassembly, min=1.0)
        scaling_factors = 1.0 / num_disassembly
        return num_disassembly, scaling_factors

    def combine_fc_weight(self, fcs):
        fc_weight = []
        if not isinstance(fcs, list):
            fcs = [fcs]
        for fc in fcs:
            fc_weight.append(fc.weight)
        fc_weight = torch.cat(fc_weight, dim=0)
        return fc_weight

    def find_similar_channels_with_scores(self, scores, r):
        # node max, node_idx shape: [C/2], index of b
        # Draw one edge from each token in A to its most similar token in B.
        node_min, node_idx = scores.min(dim=-1)
        # edge_idx shape: [C/2]
        # Keep the r most similar edges. index of a
        edge_idx = node_min.argsort(dim=-1, descending=False)

        # unm_idx shape: [C/2 -r]
        # unm_idx = edge_idx[r:]  # Unassembled Channels
        # src_idx shape: [r]
        src_idx = edge_idx[:r]  # Assembled Channels
        dst_idx = node_idx[src_idx]
        return src_idx, dst_idx, scores[src_idx, dst_idx]

    def get_channels_distance_with_weight(self, x, fc_weight):
        B, N, C = x.shape
        x = x.view(B * N, C)

        x = x.float()
        scores = bipartite_soft_matching_x_w_scores(x, fc_weight)
        return scores

    def get_masked_score(self, scores, num_disassembly):
        split_mask = num_disassembly > 1.0
        split_mask_a = split_mask[::2]
        split_index_a = split_mask_a.nonzero().squeeze()

        split_mask_b = split_mask[1::2]
        split_index_b = split_mask_b.nonzero().squeeze()

        masked_score = scores.index_fill(
            dim=0, index=split_index_a, value=torch.finfo(scores.dtype).max
        )
        masked_score = masked_score.index_fill(
            dim=1, index=split_index_b, value=torch.finfo(scores.dtype).max
        )
        return masked_score

    def find_similar_channels(self, x, fcs, num_merged_channels, scaling_factors):
        B, N, C = x.shape
        x = x.view(B * N, C)

        fc_weight = self.combine_fc_weight(fcs)

        x = x.float()
        src_idx, dst_idx, scores = bipartite_soft_matching_x_w(
            x, fc_weight, num_merged_channels, scaling_factors
        )
        return src_idx, dst_idx, scores

    def find_outlier_channels_and_store(self, x_max, th):
        with torch.no_grad():
            outlier_channel_idx = (x_max > th).nonzero().view(-1)
            num_disassembly, scaling_factors = self.find_outlier_channels(x_max, th)
            scaling_factors = scaling_factors.repeat_interleave(num_disassembly.int())
            if len(outlier_channel_idx) != 0:
                del self.outlier_channel_idx
                del self.num_disassembly
                del self.scaling_factors
                self.register_buffer("outlier_channel_idx", outlier_channel_idx)
                self.register_buffer("num_disassembly", num_disassembly)
                self.register_buffer("scaling_factors", scaling_factors)

    def find_similar_channels_and_store(self, x, fcs):
        src_idx, dst_idx, scores = self.find_similar_channels(
            x, fcs, self.num_additional_channels, self.scaling_factors
        )
        del self.src_idx
        del self.dst_idx
        print("Score: {}".format(scores))
        self.register_buffer("src_idx", src_idx)
        self.register_buffer("dst_idx", dst_idx)
        self.have_assembled = True

    def find_threshold(
        self, x, x_max, attention_mask, position_ids, fcs, function="fc1"
    ):
        num_channels = x_max.numel()
        channelmax_max = x_max.max().item()
        channelmax_min = x_max.min().item()

        th = channelmax_max
        step_num = max(100, int(channelmax_max / 0.5))
        step = (channelmax_max - channelmax_min) / (step_num)
        previous_num_additional_channels = 0

        channel_constraint = int(num_channels * self.channel_ratio)

        weight = self.combine_fc_weight(fcs)

        min_loss = 100000
        best_th = channelmax_max
        best_num_additional_channels = 0
        losses = []
        cnt = 0

        scores = self.get_channels_distance_with_weight(x, weight)
        B, N, C = x.shape

        # we first assembly and then disassembly
        while th >= channelmax_min and cnt < step_num:
            # find outlier channels
            num_disassembly, scaling_factors = self.find_outlier_channels(x_max, th)
            num_additional_channels = num_disassembly.int().sum().item() - num_channels

            if num_additional_channels > channel_constraint:
                break

            # if the num_additional_channels does not change, continue search
            if num_additional_channels != previous_num_additional_channels:
                # first assemble x, do not assemble outlier channel
                masked_scores = self.get_masked_score(scores, num_disassembly)

                # find similar channels
                src_idx, dst_idx, _ = self.find_similar_channels_with_scores(
                    masked_scores,
                    num_additional_channels,
                )

                # assemble x
                assembled_x = self.assemble_x(
                    x, src_idx, dst_idx, num_additional_channels
                )
                # assemble weight
                assembled_weight = self.assemble_weight(src_idx, dst_idx, weight)
                # assemble num_disassembly
                assembled_num_disassembly = self.assemble_vector(
                    num_disassembly, src_idx, dst_idx, num_additional_channels
                )
                # assemble scaling factor
                assembled_scaling_factors = self.assemble_vector(
                    scaling_factors, src_idx, dst_idx, num_additional_channels
                )

                # split x
                split_x = self.disassemble_x(
                    assembled_x,
                    assembled_num_disassembly,
                    assembled_scaling_factors.repeat_interleave(
                        assembled_num_disassembly.int()
                    ),
                )
                # split weight
                splited_weight = self.split_weight(
                    assembled_num_disassembly, assembled_weight
                )
                # split channel idx

                loss = self.compute_loss(
                    x,
                    split_x,
                    weight,
                    splited_weight,
                    attention_mask,
                    position_ids,
                    function,
                )
                losses.append(loss)
                if loss < min_loss:
                    min_loss = loss
                    best_th = th
                    best_num_additional_channels = num_additional_channels
                previous_num_additional_channels = num_additional_channels
            cnt += 1
            th -= step

            if cnt % 10 == 0:
                print("{} loss at iter {}".format(min_loss, cnt))
        fig = plt.figure()
        axis_x = list(range(len(losses)))
        plt.plot(axis_x, losses)
        plt.show()
        plt.close()
        print(
            "Find threshold {} by minimizing MSE with additional {} channels".format(
                best_th, best_num_additional_channels
            )
        )
        return best_th

    def qkv_function(self, x, weights, attention_mask, position_ids):
        x = x.to(weights.dtype)
        bsz, q_len, _ = x.size()
        q_output = self.layer.num_heads * self.layer.head_dim
        kv_output = self.layer.num_key_value_heads * self.layer.head_dim
        q_weight = weights[:q_output, :]
        k_weight = weights[q_output : q_output + kv_output, :]
        v_weight = weights[q_output + kv_output :, :]

        query_states = F.linear(x, q_weight)
        key_states = F.linear(x, k_weight)
        value_states = F.linear(x, v_weight)

        query_states = query_states.view(
            bsz, q_len, self.layer.num_heads, self.layer.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.layer.num_key_value_heads, self.layer.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.layer.num_key_value_heads, self.layer.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.layer.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.layer.num_key_value_groups)
        value_states = repeat_kv(value_states, self.layer.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.layer.head_dim)

        if attn_weights.size() != (bsz, self.layer.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.layer.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output

    def qkv_qfunction(self, x, weights, attention_mask, position_ids):
        x = x.to(weights.dtype)
        bsz, q_len, _ = x.size()
        q_output = self.layer.num_heads * self.layer.head_dim
        kv_output = self.layer.num_key_value_heads * self.layer.head_dim
        q_weight = weights[:q_output, :]
        k_weight = weights[q_output : q_output + kv_output, :]
        v_weight = weights[q_output + kv_output :, :]

        query_states = F.linear(x, q_weight)
        key_states = F.linear(x, k_weight)
        value_states = F.linear(x, v_weight)

        query_states = query_states.view(
            bsz, q_len, self.layer.num_heads, self.layer.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.layer.num_key_value_heads, self.layer.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.layer.num_key_value_heads, self.layer.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        cos, sin = self.layer.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        key_states = repeat_kv(key_states, self.layer.num_key_value_groups)
        value_states = repeat_kv(value_states, self.layer.num_key_value_groups)

        # use_act_quant = self.layer.qkt_matmul.use_act_quant
        # self.layer.qkt_matmul.use_act_quant = True
        # query_states = self.layer.qkt_matmul.quant_x1(query_states)
        # key_states = self.layer.qkt_matmul.quant_x2(key_states)
        # self.layer.qkt_matmul.use_act_quant = use_act_quant

        # attn_weights = self.layer.qkt_matmul(
        #     query_states, key_states.transpose(2, 3)
        # ) / math.sqrt(self.layer.head_dim)
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.layer.head_dim)

        if attn_weights.size() != (bsz, self.layer.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.layer.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        # use_act_quant = self.layer.pv_matmul.use_act_quant
        # self.layer.pv_matmul.use_act_quant = True
        # attn_weights = self.layer.pv_matmul.quant_x1(attn_weights)
        # value_states = self.layer.pv_matmul.quant_x2(value_states)
        # self.layer.pv_matmul.use_act_quant = use_act_quant
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output

    def fc_function(self, x, weights):
        out = F.linear(x.to(weights.dtype), weights)
        return out

    def post_attn_fc(self, x, weights):
        B, N, C = x.shape
        Cout, C = weights.shape
        half_Cout = Cout // 2
        out = self.fc_function(x, weights).reshape(B, N, Cout)
        x1 = out[..., :half_Cout]
        x2 = out[..., half_Cout:]
        x1 = self.act_fn(x1)
        out = x1 * x2
        return out

    def compute_output(self, x, weights, attention_mask, position_ids, function="qkv"):
        if function == "qkv":
            output = self.qkv_function(
                x, weights, attention_mask.repeat(x.shape[0], 1, 1, 1), position_ids
            )
        elif function == "fc1":
            output = self.fc_function(x, weights)
        elif function == "post_attn_norm":
            output = self.post_attn_fc(x, weights)
        else:
            raise NotImplementedError
        return output

    def compute_qoutput(self, x, weights, attention_mask, position_ids, function="qkv"):
        if function == "qkv":
            x = self.layer.q_proj.act_quantizer(x)
            weights = self.layer.q_proj.weight_quantizer(weights)
            output = self.qkv_qfunction(
                x, weights, attention_mask.repeat(x.shape[0], 1, 1, 1), position_ids
            )
        elif function == "fc1":
            x = self.layer.act_quantizer(x)
            weights = self.layer.weight_quantizer(weights)
            output = self.fc_function(x, weights)
        elif function == "post_attn_norm":
            x = self.layer.act_quantizer(x)
            weights = self.layer.weight_quantizer(weights)
            output = self.post_attn_fc(x, weights)
        else:
            raise NotImplementedError
        return output

    def compute_loss(
        self,
        ori_x,
        processed_x,
        ori_weights,
        processed_weights,
        attention_mask,
        position_ids,
        function="qkv",
    ):
        ori_output = self.compute_output(
            ori_x, ori_weights, attention_mask, position_ids, function
        )
        qoutput = self.compute_qoutput(
            processed_x, processed_weights, attention_mask, position_ids, function
        )
        mse_loss = F.mse_loss(ori_output, qoutput).item()
        return mse_loss

    def disassemble_x(self, x, num_disassembly, scaling_factors):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        B, N, C = x.shape
        x = x.view(B * N, C)
        x = torch.repeat_interleave(x, num_disassembly.int(), dim=1)
        x = x * scaling_factors.unsqueeze(0)
        C = x.shape[1]
        x = x.view(B, N, C)
        return x

    def assemble_x(self, x, src_idx, dst_idx, num_merged_channels, size=None):
        B, N, C = x.shape
        if size is None:
            size = torch.ones_like(x[0, 0])
            size = size.view(1, 1, C)

        x = assembly(
            x * size,
            src_idx,
            dst_idx,
            num_merged_channels,
            mode="sum",
        )
        size = assembly(
            size,
            src_idx,
            dst_idx,
            num_merged_channels,
            mode="sum",
        )
        x = x / size
        return x

    def split_weight(self, num_disassembly, weight):
        split_weight = torch.repeat_interleave(
            weight.data, num_disassembly.int(), dim=1
        )
        return split_weight

    def assemble_weight(self, src_idx, dst_idx, weight):
        cout, cin = weight.shape
        dev = weight.device
        dtype = weight.dtype
        ori_src_idx = torch.arange(0, cin, 2, device=dev)
        ori_dst_idx = torch.arange(1, cin, 2, device=dev)
        src_idx_ = ori_src_idx[src_idx]
        dst_idx_ = ori_dst_idx[dst_idx]
        r = src_idx_.nelement()

        channel_mask = torch.ones(cin, device=dev, dtype=dtype)
        channel_mask[src_idx_] = 0.0

        src_weight = weight.gather(dim=-1, index=src_idx_.expand(cout, r))
        assembled_weight = weight.data.scatter_reduce(
            dim=-1, index=dst_idx_.expand(cout, r), src=src_weight, reduce="sum"
        )
        assembled_weight.data = assembled_weight.data.index_select(
            -1, (channel_mask != 0).nonzero().squeeze()
        )
        return assembled_weight

    def assemble_vector(self, vector, src_idx, dst_idx, num_merged_channels):
        C = vector.shape[0]
        vector = vector.reshape(1, 1, C)
        return self.assemble_x(vector, src_idx, dst_idx, num_merged_channels).squeeze()

    def forward(self, x, size=None):
        if self.outlier_channel_idx is not None:
            if x.ndim == 2:
                x = x.unsqueeze(0)
            B, N, C = x.shape
            x = x.view(B * N, C)
            x = torch.repeat_interleave(
                x, self.num_disassembly.int().to(x.device), dim=1
            )
            x = x * self.scaling_factors.unsqueeze(0).to(x.device)
            C = x.shape[1]
            x = x.view(B, N, C)

        # only perform assembly after find_similar_channels
        if self.have_assembled:
            B, N, C = x.shape
            if size is None:
                size = torch.ones_like(x[0, 0])
                size = size.view(1, 1, C)

            x = assembly(
                x * size,
                self.src_idx.to(x.device),
                self.dst_idx.to(x.device),
                self.num_additional_channels,
                mode="sum",
            )
            size = assembly(
                size,
                self.src_idx.to(x.device),
                self.dst_idx.to(x.device),
                self.num_additional_channels,
                mode="sum",
            )
            x = x / size
        return x
