#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field 
from itertools import chain
from typing import Optional, Dict, Any, List, Tuple

import datasets
import evaluate
import torch
import torch.nn as nn
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
    PreTrainedModel
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from mla_model.configuration_qwen2_mla import Qwen2MLAConfig
from mla_model.modeling_qwen2_mla import Qwen2ForCausalLM


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

AutoConfig.register("qwen2mla", Qwen2MLAConfig)
AutoModelForCausalLM.register(Qwen2MLAConfig, Qwen2ForCausalLM)


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.49.0.dev0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)





@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def model_summary(
    model: PreTrainedModel, max_depth: int = 4, show_input_size: bool = False
) -> None:
    """
    Prints an accurate summary of the model, avoiding double-counting of parameters.

    :param PreTrainedModel model: torch model to summarize
    :param int max_depth: maximum depth of the model to print, defaults to 4
    :param bool show_input_size: whether to show input size for each layer, defaults to False
    """

    def format_params(num_params: int) -> str:
        return f"{num_params:,}" if num_params > 0 else "--"

    def format_size(size: Optional[List[int]]) -> str:
        return "x".join(str(x) for x in size) if size else "N/A"

    def count_parameters(module: nn.Module) -> Tuple[int, int]:
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(
            p.numel() for p in module.parameters() if p.requires_grad
        )
        return total_params, trainable_params

    def recursive_summarize(
        module: nn.Module, depth: int, idx: List[int], prefix: str = ""
    ) -> List[Tuple[str, int, int, int, Optional[List[int]], nn.Module]]:
        summary = []

        total_params, trainable_params = count_parameters(module)

        if depth <= max_depth:
            layer_name = f"{prefix}{type(module).__name__}"
            layer_index = ".".join(map(str, idx))
            param_shape = next(
                (p.shape for p in module.parameters(recurse=False) if p.requires_grad),
                None,
            )
            summary.append(
                (layer_name, depth, total_params, trainable_params, param_shape, module)
            )

            for i, (name, child) in enumerate(module.named_children(), 1):
                child_summary = recursive_summarize(
                    child, depth + 1, idx + [i], prefix + "  "
                )
                summary.extend(child_summary)

        return summary

    summary = recursive_summarize(model, 1, [1])

    max_name_length = max(len(name) for name, _, _, _, _, _ in summary)
    max_shape_length = max(len(format_size(shape)) for _, _, _, _, shape, _ in summary)

    print("=" * (max_name_length + 50))
    header = f"{'Layer (type:depth-idx)':<{max_name_length}} {'Output Shape':>{max_shape_length}} {'Param #':>12} {'Trainable':>10}"
    print(header)
    print("=" * (max_name_length + 50))

    for name, depth, num_params, trainable_params, shape, _ in summary:
        shape_str = format_size(shape) if show_input_size else ""
        print(
            f"{name:<{max_name_length}} {shape_str:>{max_shape_length}} {format_params(num_params):>12} {str(trainable_params > 0):>10}"
        )

    total_params, trainable_params = count_parameters(model)
    print("=" * (max_name_length + 50))
    print(f"Total params: {format_params(total_params)}")
    print(f"Trainable params: {format_params(trainable_params)}")
    print(f"Non-trainable params: {format_params(total_params - trainable_params)}")
    print("=" * (max_name_length + 50))

# Add these imports at the top of your script
from transformers import TrainerCallback, TrainerState, TrainerControl
import wandb

class GradientMonitorCallback(TrainerCallback):
    """
    A custom callback that logs gradient norms of model parameters to W&B and prints them.
    """
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        logger.info("GradientMonitorCallback: Training has started.")
        # Initialize W&B if not already initialized
        if not wandb.run:
            wandb.init(project="your_project_name", config=vars(args))
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        model = kwargs.get('model')
        if model is None:
            logger.warning("GradientMonitorCallback: No model found in kwargs.")
            return
        
        # Select parameters to monitor (e.g., first few layers)
        # Adjust the selection based on your model's architecture
        monitored_params = {
            name: param for name, param in model.named_parameters() if 'self_attn' in name or 'mlp' in name
        }
        
        # Calculate and log gradient norms
        for name, param in monitored_params.items():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                wandb.log({f"grad_norm/{name}": grad_norm})
                # Optionally, print to console
                if state.global_step % 100 == 0:
                    logger.info(f"Gradient norm for {name}: {grad_norm:.4f}")
        
        # Optionally, log total gradient norm
        total_grad_norm = sum(param.grad.norm(2).item() ** 2 for param in monitored_params.values() if param.grad is not None) ** 0.5
        wandb.log({"grad_norm/total": total_grad_norm})
        if state.global_step % 100 == 0:
            logger.info(f"Total gradient norm: {total_grad_norm:.4f}")


import inspect
import logging

from functools import partial
from typing import Callable

import transformers

from packaging import version
from transformers import PreTrainedModel

from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.transformers.functional import liger_cross_entropy
from liger_kernel.transformers.geglu import LigerGEGLUMLP
from liger_kernel.transformers.layer_norm import LigerLayerNorm
from liger_kernel.transformers.model.gemma import lce_forward as gemma_lce_forward
from liger_kernel.transformers.model.gemma import lce_forward_deprecated as gemma_lce_forward_deprecated
from liger_kernel.transformers.model.gemma2 import lce_forward as gemma2_lce_forward
from liger_kernel.transformers.model.gemma2 import lce_forward_deprecated as gemma2_lce_forward_deprected
from liger_kernel.transformers.model.llama import lce_forward as llama_lce_forward
from liger_kernel.transformers.model.llama import lce_forward_deprecated as llama_lce_forward_deprecated
from liger_kernel.transformers.model.mistral import lce_forward as mistral_lce_forward
from liger_kernel.transformers.model.mixtral import lce_forward as mixtral_lce_forward
from liger_kernel.transformers.model.mixtral import lce_forward_deprecated as mixtral_lce_forward_deprecated
from liger_kernel.transformers.model.phi3 import lce_forward as phi3_lce_forward
from liger_kernel.transformers.model.phi3 import lce_forward_deprecated as phi3_lce_forward_deprecated
from liger_kernel.transformers.model.qwen2 import lce_forward as qwen2_lce_forward
from liger_kernel.transformers.model.qwen2 import lce_forward_deprecated as qwen2_lce_forward_deprecated
from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from liger_kernel.transformers.swiglu import LigerBlockSparseTop2MLP
from liger_kernel.transformers.swiglu import LigerPhi3SwiGLUMLP
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

from liger_kernel.transformers.monkey_patch import _bind_method_to_module, _patch_rms_norm_module

transformer_version = version.parse(transformers.__version__)

logger = logging.getLogger(__name__)
SUPPORTED_TRANSFORMER_VERSION = "4.46.1"
TRANSFORMER_DEPRECATION_WARNING = "Support for transformers versions < 4.46.1 will soon be discontinued due to issues with incorrect gradient accumulation. \n Please consider upgrading to avoid potential issues. See details: https://github.com/huggingface/transformers/pull/34191"


def apply_liger_kernel_to_qwen2_mla(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2 models

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from mla_model import modeling_qwen2_mla

    from mla_model.modeling_qwen2_mla import Qwen2Model



    if rope:
        modeling_qwen2_mla.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_qwen2_mla.Qwen2RMSNorm = LigerRMSNorm

    if cross_entropy:
        if transformer_version >= version.parse(SUPPORTED_TRANSFORMER_VERSION):
            from transformers.loss.loss_utils import nn

            nn.functional.cross_entropy = liger_cross_entropy
        else:
            logger.warning(TRANSFORMER_DEPRECATION_WARNING)
            modeling_qwen2_mla.CrossEntropyLoss = LigerCrossEntropyLoss

    if fused_linear_cross_entropy:
        if transformer_version >= version.parse(SUPPORTED_TRANSFORMER_VERSION):
            modeling_qwen2_mla.Qwen2ForCausalLM.forward = qwen2_lce_forward
        else:  # if version < 4.46.1
            logger.warning(TRANSFORMER_DEPRECATION_WARNING)
            modeling_qwen2_mla.Qwen2ForCausalLM.forward = qwen2_lce_forward_deprecated

    if swiglu:
        modeling_qwen2_mla.Qwen2MLP = LigerSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: Qwen2Model = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)

        for decoder_layer in base_model.layers:
            if swiglu:
                _bind_method_to_module(decoder_layer.mlp, "forward", LigerSwiGLUMLP.forward)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)
    print("Applied Liger kernels to Qwen2")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()



    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    
    # Set seed before initializing model.
    set_seed(training_args.seed)


    raw_datasets = datasets.load_dataset("Zyphra/Zyda-2", name="sample-100BT", split="train", streaming=True)

    def freeze_non_attention_layers(model):
        """
        Freezes all parameters except attention-related layers in a Qwen2T6 model.
        Specifically handles T6's specialized attention architecture including CPLinear 
        and Cross-Layer Attention (CLA) components.
        
        Args:
            model: A Qwen2T6 model instance
        """
        # T6-specific attention components
        attention_keywords = [
            'self_attn',           # Main attention module
            'q_proj',              # Query projection
            'W_A',                 # CP decomposition weights for keys
            'W_B',                 # CP decomposition weights for values
            'o_proj',              # Output projection
            'rotary_emb',          # Rotary embeddings
        ]
        
        def is_attention_parameter(name):
            return any(keyword in name for keyword in attention_keywords)
        
        # Print initial parameter status
        print("Starting parameter freezing process...")
        print("\nAttention-related components that will remain trainable:")
        
        for name, param in model.named_parameters():
            is_attn = is_attention_parameter(name)
            param.requires_grad = is_attn
            
            if is_attn:
                print(f"- {name}")
                
        # Calculate and print statistics
        total_params = 0
        trainable_params = 0
        
        # Group parameters by component type
        component_stats = {
            'attention': 0,
            'mlp': 0,
            'norm': 0,
            'embedding': 0,
            'other': 0
        }
        
        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            
            if param.requires_grad:
                trainable_params += num_params
            
            # Categorize parameters
            if 'self_attn' in name or 'W_A' in name or 'W_B' in name or 'q_proj' in name or 'o_proj' in name:
                component_stats['attention'] += num_params
            elif 'mlp' in name:
                component_stats['mlp'] += num_params
            elif 'norm' in name:
                component_stats['norm'] += num_params
            elif 'embed' in name:
                component_stats['embedding'] += num_params
            else:
                component_stats['other'] += num_params
        
        # Print detailed statistics
        print("\nParameter Statistics:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
        
        print("\nParameter Distribution:")
        for component, count in component_stats.items():
            percentage = 100 * count / total_params
            print(f"{component.capitalize()}: {count:,} ({percentage:.2f}%)")
        
        print("\nNote: Only attention-related components are trainable, including:")
        print("- Query projections")
        print("- CP decomposition weights (W_A, W_B)")
        print("- Output projections")

    def load_and_prepare_qwen2t6_model(
        pretrained_model_name: str,
        custom_config: Dict[str, Any] = None
    ) -> torch.nn.Module:
        """
        Load a pretrained Qwen2 model and prepare it for training with frozen MLP layers.
        
        Args:
            pretrained_model_name: Name or path of the pretrained Qwen2 model
            custom_config: Optional custom configuration for Qwen2T6
        """
        # Load pretrained model and config
        pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
        pretrained_config = pretrained_model.config
        
        # Create Qwen2T6 config by combining pretrained and custom configs
        t6_config = Qwen2MLAConfig(
            **{k: v for k, v in pretrained_config.to_dict().items()}
        )
        t6_config.num_key_value_heads=14

        
        # Initialize your custom model with the combined config
        custom_model = AutoModelForCausalLM.from_config(t6_config, torch_dtype=torch.bfloat16,      
                                        # attn_implementation="flash_attention_2")
                                        attn_implementation="sdpa")
        
        # Transfer weights from pretrained to custom model
        transfer_weights(pretrained_model, custom_model)
        
        # Freeze MLP layers
        # freeze_mlp_layers(custom_model)
        freeze_non_attention_layers(custom_model)
        
        return custom_model

    def transfer_weights(src_model: torch.nn.Module, dst_model: torch.nn.Module):
        """Transfer matching weights from source to destination model."""
        src_state_dict = src_model.state_dict()
        dst_state_dict = dst_model.state_dict()
        
        # Filter and transfer matching parameters
        transferred_state_dict = {}
        for name, param in dst_state_dict.items():
            if name in src_state_dict:
                # Handle special cases for CP decomposed layers
                if 'k_proj' in name or 'v_proj' in name:
                    # Skip CP decomposed layers - they'll be initialized with their own parameters
                    continue
                transferred_state_dict[name] = src_state_dict[name]
        
        # Load transferred weights
        dst_model.load_state_dict(transferred_state_dict, strict=False)

    def freeze_mlp_layers(model: torch.nn.Module):
        """Freeze all MLP layers in the model."""
        for name, param in model.named_parameters():
            # Freeze parameters in MLP layers
            if any(x in name for x in ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']):
                param.requires_grad = False

    def count_trainable_parameters(model: torch.nn.Module) -> tuple:
        """Count trainable and total parameters in the model."""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        return trainable_params, total_params
    
    model = load_and_prepare_qwen2t6_model(
        pretrained_model_name=model_args.model_name_or_path,
        custom_config=    {
        "use_cla": True,
        "cla_share_factor": 2,}
    )

    apply_liger_kernel_to_qwen2_mla(model=model)

    # Print model summary
    model_summary(model, max_depth=10)

    # Count parameters
    trainable_params, total_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )



    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))


    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples['text'], truncation=True, max_length=2048, padding="max_length")
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):

        tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=["text", "nemo_id"],
            )
        
    max_pos_embeddings = 2048

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > max_pos_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            if max_pos_embeddings > 0:
                block_size = min(1024, max_pos_embeddings)
            else:
                block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)


    

    def group_texts(examples):
        """
        Takes in a dictionary of lists and adds labels for causal language modeling.
        """
        # Copy input_ids to create labels
        examples["labels"] = examples["input_ids"].copy()
        return examples

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    with training_args.main_process_first(desc="grouping texts together"):

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
        )
        
    if training_args.do_eval:

        dataset = load_dataset("BEE-spoke-data/QualiMix-01", "deduped")

        eval_dataset = dataset['test'].map(
                tokenize_function,
                batched=True,
                remove_columns=dataset['test'].column_names,
            ).map(
            group_texts,
            batched=True,
        )
        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)
    
    if training_args.do_train:
        train_dataset = lm_datasets
   

    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=2048
    )
            
    training_args.save_safetensors=False 

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
        # data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_xla_available()
        else None,
            callbacks=[
                # GradientMonitorCallback()
                ]
    )

    # Training
    if training_args.do_train:
        
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    import debugpy

    debugpy.listen(5678)
    debugpy.wait_for_client()
    main()