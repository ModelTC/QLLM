import logging
import os
import sys
import time

# from torch._six import inf
from math import inf

import torch
from termcolor import colored


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
        retain_graph=False,
    ):
        self._scaler.scale(loss).backward(
            create_graph=create_graph, retain_graph=retain_graph
        )
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def create_logger(output_dir, dist_rank=0, name=""):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
    color_fmt = (
        colored("[%(asctime)s %(name)s]", "green")
        + colored("(%(filename)s %(lineno)d)", "yellow")
        + ": %(levelname)s %(message)s"
    )

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(
        os.path.join(output_dir, f"log_rank{dist_rank}_{int(time.time())}.txt"),
        mode="a",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    return logger


oc_maxmin_dict = {}


def layer_omax_hook(m, i, o):
    name = m.name
    if not isinstance(o, torch.Tensor):
        return
    if o.ndim == 3:
        xmax = torch.amax(o, [0, 1])  # shape d
        xmin = torch.amin(o, [0, 1])  # shape d
    elif o.ndim == 2:
        xmax = torch.amax(o, [0])  # shape d
        xmin = torch.amin(o, [0])  # shape d
    # if "q_proj" in name or "k_proj" in name:
    #     o_shape = o.shape
    #     reshape_o = o.reshape(o_shape[0], o_shape[1], m.num_heads, m.head_dim).transpose(1, 2)
    #     xmax = torch.amax(reshape_o, [0, 1, 2])
    #     xmin = torch.amin(reshape_o, [0, 1, 2])

    if name not in oc_maxmin_dict:
        oc_maxmin_dict[name] = (xmax.detach_(), xmin.detach_())
    else:
        oc_maxmin_dict[name] = (
            torch.max(oc_maxmin_dict[name][0], xmax).detach_(),
            torch.min(oc_maxmin_dict[name][1], xmin).detach_(),
        )
        # oc_maxmin_dict[name] = oc_maxmin_dict[name][0]*0.99+xmax*0.01,oc_maxmin_dict[name][1]*0.99+xmin*0.01


oc_mean_std_dict = {}


def layer_omean_std_hook(m, i, o):
    name = m.name
    if not isinstance(o, torch.Tensor):
        return
    std, mean = torch.std_mean(o)

    if name not in oc_mean_std_dict:
        oc_mean_std_dict[name] = (mean.detach_(), std.detach_())
    else:
        oc_mean_std_dict[name] = (
            (oc_mean_std_dict[name][0] * 0.99 + mean).detach_(),
            (oc_mean_std_dict[name][1] * 0.99 + std).detach_(),
        )


ic_maxmin_dict = {}


def layer_i0max_hook(m, i, o):
    name = m.name
    if len(i) == 0 or not isinstance(i[0], torch.Tensor):
        return
    if i[0].ndim == 3:
        xmax = torch.amax(i[0], [0, 1])  # shape d
        xmin = torch.amin(i[0], [0, 1])  # shape d
    elif i[0].ndim == 2:
        xmax = torch.amax(i[0], [0])  # shape d
        xmin = torch.amin(i[0], [0])  # shape d
    elif i[0].ndim == 4:
        xmax = torch.amax(i[0], [0, 1, 2])
        xmin = torch.amin(i[0], [0, 1, 2])

    if name not in ic_maxmin_dict:
        ic_maxmin_dict[name] = xmax.detach_(), xmin.detach_()
    else:
        ic_maxmin_dict[name] = (
            torch.max(ic_maxmin_dict[name][0], xmax).detach_(),
            torch.min(ic_maxmin_dict[name][1], xmin).detach_(),
        )
        # ic_maxmin_dict[name] = ic_maxmin_dict[name][0]*0.99+xmax*0.01,ic_maxmin_dict[name][1]*0.99+xmin*0.01


def layer_i01max_hook(m, i, o):
    name = m.name
    if len(i) == 0 or not isinstance(i[0], torch.Tensor):
        return
    if i[0].ndim == 3:
        i0_xmax = torch.amax(i[0], [0, 1])  # shape d
        i0_xmin = torch.amin(i[0], [0, 1])  # shape d
        i1_xmax = torch.amax(i[1], [0, 1])  # shape d
        i1_xmin = torch.amin(i[1], [0, 1])  # shape d
    elif i[0].ndim == 2:
        i0_xmax = torch.amax(i[0], [0])  # shape d
        i0_xmin = torch.amin(i[0], [0])  # shape d
        i1_xmax = torch.amax(i[1], [0])  # shape d
        i1_xmin = torch.amin(i[1], [0])  # shape d

    # for i0
    i0_name = f"{name}_i0"
    if i0_name not in ic_maxmin_dict:
        ic_maxmin_dict[i0_name] = i0_xmax.detach_(), i0_xmin.detach_()
    else:
        ic_maxmin_dict[i0_name] = (
            torch.max(ic_maxmin_dict[i0_name][0], i0_xmax).detach_(),
            torch.min(ic_maxmin_dict[i0_name][1], i0_xmin).detach_(),
        )
        # ic_maxmin_dict[name] = ic_maxmin_dict[name][0]*0.99+xmax*0.01,ic_maxmin_dict[name][1]*0.99+xmin*0.01

    i1_name = f"{name}_i1"
    if i1_name not in ic_maxmin_dict:
        ic_maxmin_dict[i1_name] = i1_xmax.detach_(), i1_xmin.detach_()
    else:
        ic_maxmin_dict[i1_name] = (
            torch.max(ic_maxmin_dict[i1_name][0], i1_xmax).detach_(),
            torch.min(ic_maxmin_dict[i1_name][1], i1_xmin).detach_(),
        )


oc_mean_feat_dict = {}
oc_mean_feat_num_dict = {}


def layer_omean_feature_hook(m, i, o):
    name = m.name
    if not isinstance(o, torch.Tensor):
        return

    o = o.float()

    if name not in oc_mean_feat_dict:
        oc_mean_feat_dict[name] = o.detach_()
        oc_mean_feat_num_dict[name] = 1.0
    else:
        oc_mean_feat_dict[name] = (
            oc_mean_feat_dict[name] * oc_mean_feat_num_dict[name] + o.detach_()
        ) / (oc_mean_feat_num_dict[name] + 1)
        oc_mean_feat_num_dict[name] += 1.0


oc_feat_dict = {}


def layer_o_feature_hook(m, i, o):
    name = m.name
    bs = m.bs
    if not isinstance(o, torch.Tensor):
        return

    if name not in oc_feat_dict:
        oc_feat_dict[name] = o.detach_()
    else:
        if oc_feat_dict[name].shape[0] < bs:
            oc_feat_dict[name] = torch.cat([oc_feat_dict[name], o], dim=0)
        # oc_mean_feat_dict[name] = (
        #     oc_mean_feat_dict[name] * oc_mean_feat_num_dict[name] + o.detach_()
        # ) / (oc_mean_feat_num_dict[name] + 1)
        # oc_mean_feat_num_dict[name] += 1.0


oc_norm_dict = {}
oc_norm_num_dict = {}


def layer_onorm_hook(m, i, o):
    name = m.name
    if not isinstance(o, torch.Tensor):
        return
    if len(o.shape) == 2:
        o = o.unsqueeze(0)
    B, N, C = o.shape
    o = o.reshape(B * N, C)
    o = o.t()
    o = o.float()

    if name not in oc_norm_dict:
        oc_norm_dict[name] = torch.norm(o, p=2, dim=1) ** 2
        oc_norm_num_dict[name] = 1.0
    else:
        oc_norm_dict[name] = (
            oc_norm_dict[name] * oc_norm_num_dict[name] + torch.norm(o, p=2, dim=1) ** 2
        ) / (oc_norm_num_dict[name] + 1)
        oc_norm_num_dict[name] += 1.0
