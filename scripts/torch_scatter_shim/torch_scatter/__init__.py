"""
Pure-PyTorch stand-in for the parts of torch_scatter that gridfm-graphkit uses:
scatter_add, scatter_mean, scatter_max.  Same call signatures for these ops, no
compiled extension, so it works on aarch64 / Blackwell where no wheel exists.
scatter_max returns (values, argmax) like the original; argmax is only filled
for API compatibility and must not be relied on.
"""
import torch

__version__ = "2.1.2+purtorch"


def _expand(index, src, dim):
    if dim < 0:
        dim += src.dim()
    if index.dim() == 1 and src.dim() > 1:
        shape = [1] * src.dim()
        shape[dim] = -1
        index = index.view(shape)
    return index.expand_as(src), dim


def _size(index, src, dim, dim_size):
    if dim_size is not None:
        return dim_size
    return int(index.max()) + 1 if index.numel() else 0


def scatter_add(src, index, dim=-1, out=None, dim_size=None):
    index, dim = _expand(index, src, dim)
    if out is None:
        shape = list(src.shape)
        shape[dim] = _size(index, src, dim, dim_size)
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    total = scatter_add(src, index, dim, out, dim_size)
    ones = torch.ones_like(src)
    count = scatter_add(ones, index, dim, None, total.size(dim if dim >= 0 else dim + src.dim()))
    return total / count.clamp(min=1)


def scatter_max(src, index, dim=-1, out=None, dim_size=None):
    index, dim = _expand(index, src, dim)
    shape = list(src.shape)
    shape[dim] = _size(index, src, dim, dim_size)
    base = torch.zeros(shape, dtype=src.dtype, device=src.device)
    vals = base.scatter_reduce(dim, index, src, reduce="amax", include_self=False)
    return vals, torch.zeros_like(vals, dtype=torch.long)
