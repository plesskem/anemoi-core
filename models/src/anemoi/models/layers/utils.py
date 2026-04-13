# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Optional

from hydra.errors import InstantiationException
from hydra.utils import instantiate
import torch
from torch import nn
from torch import cuda
from torch.utils.checkpoint import checkpoint
from contextlib import contextmanager

from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class CheckpointWrapper(nn.Module):
    """Wrapper for checkpointing a module."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return checkpoint(self.module, *args, **kwargs, use_reentrant=False)


def maybe_checkpoint(func, enabled: bool, *args, **kwargs):
    """Conditionally apply gradient checkpointing to a function.

    Parameters
    ----------
    func : callable
        The function to potentially wrap with checkpointing
    enabled : bool
        Whether to apply gradient checkpointing
    *args, **kwargs
        Arguments to pass to the function

    Returns
    -------
    The result of calling func with the provided arguments
    """
    if enabled:
        return checkpoint(func, *args, **kwargs, use_reentrant=False)
    return func(*args, **kwargs)


def load_layer_kernels(kernel_config: Optional[DotDict] = None, instance: bool = True) -> DotDict["str" : nn.Module]:
    """Load layer kernels from the config.

    This function tries to load the layer kernels from the config. If the layer kernel is not supplied, it will fall back to the torch.nn implementation.

    Parameters
    ----------
    kernel_config : DotDict
        Kernel configuration, e.g. {"Linear": {"_target_": "torch.nn.Linear"}}
    instance : bool
        If True, instantiate the kernels. If False, return the config.
        This is useful for testing purposes.
        Defaults to True.

    Returns
    -------
    DotDict
        Container with layer factories.
    """
    # If self.layer_kernels entry is missing from the config, use torch.nn kernels
    default_kernels = {
        "Linear": {"_target_": "torch.nn.Linear"},
        "LayerNorm": {"_target_": "torch.nn.LayerNorm"},
        "Activation": {"_target_": "torch.nn.GELU"},
        "QueryNorm": {
            "_target_": "anemoi.models.layers.normalization.AutocastLayerNorm",
            "_partial_": True,
            "bias": False,
        },
        "KeyNorm": {
            "_target_": "anemoi.models.layers.normalization.AutocastLayerNorm",
            "_partial_": True,
            "bias": False,
        },
    }

    if kernel_config is None:
        kernel_config = DotDict()

    layer_kernels = DotDict()

    # Loop through all kernels in the layer_kernels config entry and try import them
    for name, kernel_entry in {**default_kernels, **kernel_config}.items():
        if instance:
            try:
                layer_kernels[name] = instantiate(kernel_entry, _partial_=True)
            except InstantiationException:
                LOGGER.info(
                    f"{kernel_entry['_target_']} not found! Check your config.model.layer_kernel. {name} entry. Maybe your desired kernel is not installed or the import string is incorrect?"
                )
                raise InstantiationException
            else:
                LOGGER.info(f"{name} kernel: {kernel_entry['_target_']}.")
        else:
            layer_kernels[name] = kernel_entry
    return layer_kernels

#class ProfilerWrapper(nn.Module):
    #"""Wrapper for checkpointing a module."""

    #def __init__(self, module: nn.Module, marker: str) -> None:
        #super().__init__()
        #self.module = module
        ##self.marker=module.__class__.__name__
        #self.marker=marker
        #self.enabled=True
        
        ## Register backward hook for profiling backward pass
        ##self.register_full_backward_hook(self._backward_hook)
       ## self.register_full_backward_pre_hook(self._backward_pre_hook)

    #def forward(self, *args, **kwargs):
        ##tracing_marker=marker.split('- ')[1].split(', input')[0]
        #with torch.autograd.profiler.record_function("anemoi-"+self.marker):
            #out = self.module(*args, **kwargs)
        #return out

def user_annotate_children(model: nn.Module, prefix: str = '', use_backward: bool = True) -> nn.Module:
    """Annotate all modules with profiler markers for forward and optionally backward.

    Forward: wraps module.forward in a record_function context manager.
    Backward: attaches tensor hooks to both INPUT and OUTPUT tensors of each module.
              The hook on the output fires first (backward is reversed), opening a range.
              The hook on the input fires second, closing the range.

    Parameters
    ----------
    model : nn.Module
        The model to annotate.
    prefix : str
        Prefix for the annotation names (builds hierarchical names).
    use_backward : bool
        If True, also annotate backward pass using paired tensor hooks.
        If False, only annotate forward with record_function context manager.

    Returns
    -------
    nn.Module
        The annotated model (modified in-place).
    """

    def _make_marker(name: str, module: nn.Module) -> str:
        parts = [part for part in name.split('.') if not part.isdigit()]
        return f"{module.__class__.__name__}-{'.'.join(parts)}"

    for child_name, child in model.named_children():
        full_name = f"{prefix}{child_name}"
        marker = _make_marker(full_name, child)

        if not hasattr(child, "_original_forward"):
            child._original_forward = child.forward

            def make_annotated_forward(mod, m=marker, annotate_bwd=use_backward):
                def annotated_forward(*args, **kwargs):
                    # Capture input tensors BEFORE forward
                    input_tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]

                    with torch.profiler.record_function(f"anemoi-{m}.forward"):
                        result = mod._original_forward(*args, **kwargs)

                    if not annotate_bwd:
                        return result

                    # Collect output tensors
                    output_tensors = []
                    if isinstance(result, torch.Tensor) and result.requires_grad:
                        output_tensors = [result]
                    elif isinstance(result, tuple):
                        output_tensors = [r for r in result if isinstance(r, torch.Tensor) and r.requires_grad]

                    if output_tensors and input_tensors:
                        # Use a mutable container to share the range across hooks
                        state = {"range": None}

                        # Output hook fires FIRST in backward (backward is reversed)
                        def output_hook(grad, _m=m, _state=state):
                            _state["range"] = torch.autograd.profiler.record_function(f"anemoi-{_m}.backward")
                            _state["range"].__enter__()
                            return grad

                        # Input hook fires SECOND in backward → close the range
                        def input_hook(grad, _m=m, _state=state):
                            if _state["range"] is not None:
                                _state["range"].__exit__(None, None, None)
                                _state["range"] = None
                            return grad

                        # Register on first output and first input only
                        output_tensors[0].register_hook(output_hook)
                        input_tensors[0].register_hook(input_hook)

                    return result
                return annotated_forward

            child.forward = make_annotated_forward(child)

        user_annotate_children(child, prefix=f"{full_name}.", use_backward=use_backward)

    return model

#def user_annotate_children(model: nn.Module, prefix: str = '') -> nn.Module:
    #def add_annotations_to_forward(name, module):
        ## ensure original forward is stored only once
        #if not hasattr(module, "_forward"):
            #module._forward = module.forward

        #def annotated_forward(*args, **kwargs):
            #parts = [part for part in name.split('.') if not part.isdigit()]
            #marker = f"{module.__class__.__name__}-{'.'.join(parts)}"
            #with torch.profiler.record_function(f"ANEMOI-{marker}.forward"):
                #return module._forward(*args, **kwargs)
        #return annotated_forward
    #for child_name, child in model.named_children():
        #child.forward = add_annotations_to_forward(f"{prefix}{child_name}", child)
        #user_annotate_children(child, prefix=f"{prefix}{child_name}.")

    ## handle leaf (no children)
    #if len(list(model.children())) == 0:
        #model.forward = add_annotations_to_forward(f"{prefix}(LEAF)", model)
    #return model
