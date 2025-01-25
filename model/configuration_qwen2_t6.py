#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file was automatically generated from src/transformers/models/qwen2_t6/modular_qwen2_t6.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. If any change should be done, please apply the change to the
#                          modular_qwen2_t6.py file directly. One of our CI enforces this.
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation


class Qwen2T6Config(PretrainedConfig):
    """
    Configuration class for Qwen2T6 model. Extends Qwen2Config with additional parameters
    for Conditional Layer Attention (CLA).

    Args:
        use_cla (`bool`, *optional*, defaults to True):
            Whether to use Conditional Layer Attention
        cla_share_factor (`int`, *optional*, defaults to 2):
            Sharing factor for Conditional Layer Attention
    """

    model_type = "qwen2t6"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen2T6`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    def __init__(self,
                    vocab_size=151936,
                    hidden_size=4096,
                    intermediate_size=22016,
                    num_hidden_layers=32,
                    num_attention_heads=32,
                    num_key_value_heads=32,
                    hidden_act="silu",
                    max_position_embeddings=32768,
                    initializer_range=0.02,
                    rms_norm_eps=1e-6,
                    use_cache=True,
                    tie_word_embeddings=False,
                    rope_theta=10000.0,
                    rope_scaling=None,
                    use_sliding_window=False,
                    sliding_window=4096,
                    max_window_layers=28,
                    attention_dropout=0.0,
                    attention_bias=False,
                    use_cla=True,
                    cla_share_factor=2,
                    cp_rank=2,
                    cp_q_rank=12,
                    use_qk_norm=True,


                    **kwargs):
        # Call parent class's __init__
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        
        # Remove the comma after tie_word_embeddings assignment
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.use_cla = use_cla
        self.cla_share_factor = cla_share_factor
        self.cp_rank = cp_rank
        self.cp_q_rank = cp_q_rank
        self.use_qk_norm = use_qk_norm