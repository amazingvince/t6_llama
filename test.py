from model.configuration_qwen2_t6 import Qwen2T6Config
from model.modeling_qwen2_t6 import Qwen2ForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any
import torch

AutoConfig.register("qwen2t6", Qwen2T6Config)
AutoModelForCausalLM.register(Qwen2T6Config, Qwen2ForCausalLM)

def load_and_prepare_qwen2t6_model(
    pretrained_model_name: str,
    custom_config: Optional[Dict[str, Any]] = None
) -> torch.nn.Module:
    """
    Load a pretrained Qwen2 model and prepare it for training with CLA and frozen MLP layers.
    Args:
        pretrained_model_name: Name or path of the pretrained Qwen2 model
        custom_config: Optional custom configuration for Qwen2T6
    """
    # Load pretrained model and config
    pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name)
    pretrained_config = pretrained_model.config
    
    # Create base Qwen2T6 config from pretrained config
    base_config = {
        k: v for k, v in pretrained_config.to_dict().items()
        if k not in ['architectures', '_name_or_path']
    }
    
    # Add T6-specific config parameters
    t6_config_updates = {
        "use_cla": True,
        "cla_share_factor": 2,
        "cp_rank": 2,
        "attention_bias": False,  # T6 doesn't use bias in attention
    }
    
    # Update with any custom config
    if custom_config:
        t6_config_updates.update(custom_config)
    
    # Combine configurations
    final_config = {**base_config, **t6_config_updates}
    t6_config = Qwen2T6Config(**final_config)
    
    # Initialize custom model
    custom_model = AutoModelForCausalLM.from_config(t6_config,  
                                    torch_dtype=torch.bfloat16,      
                                        attn_implementation="flash_attention_2").to('cuda:0')
                                        # attn_implementation="sdpa")
    
    # Transfer weights from pretrained to custom model
    transfer_weights(pretrained_model, custom_model)
    
    # Freeze MLP layers
    freeze_mlp_layers(custom_model)
    
    return custom_model

def transfer_weights(src_model: torch.nn.Module, dst_model: torch.nn.Module):
    """Transfer matching weights from source to destination model."""
    src_state_dict = src_model.state_dict()
    dst_state_dict = dst_model.state_dict()
    
    transferred_state_dict = {}
    for name, param in dst_state_dict.items():
        if name in src_state_dict:
            # For layers that use CLA, only transfer weights for layers that are multiples of cla_share_factor
            if ('k_proj' in name or 'v_proj' in name):
                layer_idx = int(name.split('.')[2])  # Assuming format like model.layers.0.k_proj
                if layer_idx % dst_model.config.cla_share_factor == 0:
                    transferred_state_dict[name] = src_state_dict[name]
            else:
                transferred_state_dict[name] = src_state_dict[name]
    
    # Load transferred weights
    dst_model.load_state_dict(transferred_state_dict, strict=False)

def freeze_mlp_layers(model: torch.nn.Module):
    """Freeze all MLP layers in the model."""
    for name, param in model.named_parameters():
        if any(x in name for x in ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']):
            param.requires_grad = False

def count_trainable_parameters(model: torch.nn.Module) -> tuple:
    """Count trainable and total parameters in the model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params

# Example usage
def main():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare model
    model = load_and_prepare_qwen2t6_model(
        pretrained_model_name="Qwen/Qwen2.5-0.5B",
        custom_config={
            "use_cla": True,
            "cla_share_factor": 2,
            "cp_rank": 2,
            "attention_bias": False,
        }
    )
    
    # Test generation
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt", padding=True)
    outputs = model.generate(
        input_ids=inputs["input_ids"].to('cuda:0'),
        # attention_mask=inputs["attention_mask"],
        max_length=50,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()