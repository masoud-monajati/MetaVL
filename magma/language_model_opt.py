
import torch
from transformers import GPTNeoForCausalLM, AutoConfig, GPT2LMHeadModel
from .utils import print_main
from pathlib import Path
from transformers.modeling_utils import no_init_weights

from transformers import GPT2Tokenizer, GPT2Model

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, OPTForCausalLM

from transformers import OPTModel, OPTConfig

#tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
#from .meta.MetaICL.metaicl.data import MetaICLData
#from .meta.MetaICL.metaicl.model import MetaICLModel

# Load the model
#data = MetaICLData(method="channel", max_length=1024, max_length_per_example=256)



LANGUAGE_MODELS = [
    "gpt2",
]
'''
GPT2Config {
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "gradient_checkpointing": false,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 1024,
  "n_head": 16,
  "n_inner": null,
  "n_layer": 24,
  "n_positions": 1024,
  "n_special": 0,
  "predict_special_tokens": true,
  "resid_pdrop": 0.1,
  "scale_attn_weights": true,
  "summary_activation": null,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": true,
  "summary_type": "cls_index",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 50
    }
  },
  "transformers_version": "4.6.0.dev0",
  "use_cache": true,
  "vocab_size": 50257
}

6.7 OPTConfig {
  "_name_or_path": "facebook/opt-6.7b",
  "_remove_final_layer_norm": false,
  "activation_dropout": 0.0,
  "activation_function": "relu",
  "architectures": [
    "OPTForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 2,
  "do_layer_norm_before": true,
  "dropout": 0.1,
  "eos_token_id": 2,
  "ffn_dim": 16384,
  "hidden_size": 4096,
  "init_std": 0.02,
  "layerdrop": 0.0,
  "max_position_embeddings": 2048,
  "model_type": "opt",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "pad_token_id": 1,
  "prefix": "</s>",
  "torch_dtype": "float16",
  "transformers_version": "4.20.1",
  "use_cache": true,
  "vocab_size": 50272,
  "word_embed_proj_dim": 4096
}

'''

def opt_config27():
    config = OPTConfig()
    config.word_embed_proj_dim = 2560
    config.torch_dtype = "float16"
    config.prefix = "</s>"
    config.num_hidden_layers = 32
    config.num_attention_heads = 32
    config.hidden_size = 2560
    config.ffn_dim = 10240
    config.architectures = [
    "OPTForCausalLM"
  ]
    config.jax = True
    config.gradient_checkpointing = True
    return config

def opt_config67():
    config = OPTConfig()
    config.word_embed_proj_dim = 4096
    config.torch_dtype = "float16"
    config.prefix = "</s>"
    config.num_hidden_layers = 32
    config.num_attention_heads = 32
    config.hidden_size = 4096
    config.ffn_dim = 16384
    config.architectures = [
    "OPTForCausalLM"
  ]
    config.jax = True
    config.gradient_checkpointing = True
    return config


def opt_config():
    config = OPTConfig()
    config.word_embed_proj_dim = 2048
    config.torch_dtype = "float16"
    config.prefix = "</s>"
    config.num_hidden_layers = 24
    config.num_attention_heads = 32
    config.hidden_size = 2048
    config.ffn_dim = 8192
    config.architectures = [
    "OPTForCausalLM"
  ]
    config.jax = True
    config.gradient_checkpointing = True
    return config


def get_gptj(
    gradient_checkpointing: bool = True,
    from_pretrained=False,
) -> torch.nn.Module:
    """
    Loads GPTJ language model from HF
    """
    print_main("Loading GPTJ language model...")
    '''
    #config = opt_config()
    config = opt_config27()
    #config = opt_config67()
    print('original',config)
    config1 = AutoConfig.from_pretrained("facebook/opt-2.7b")
    print("2.7",config1)
    #print(begh)
    
    #config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
    #print(config)
    #print(begh)
    config.gradient_checkpointing = gradient_checkpointing
    if gradient_checkpointing:
        config.use_cache = False
    #config.model_device = "cpu"
    '''
    if from_pretrained:
        raise NotImplemented("GPTJ pretrained not implemented")
    else:
        #with no_init_weights():
        print('loading...')
        #model = AutoModelForCausalLM.from_config(config)
        #model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
        model = OPTForCausalLM.from_pretrained("facebook/opt-1.3b")
        model.config.gradient_checkpointing = True
        model.config.use_cache = False
        model.config.model_device = "cpu"
        '''
        torch.save(model.state_dict(),"/home/monajati/main/metaVL/magma/gpt/opt-2.7b.pt")
        #model.save_pretrained(save_directory = "/gpt/")
        model1 = OPTForCausalLM(config1)
        state_dict = torch.load("/home/monajati/main/metaVL/magma/gpt/opt-2.7b.pt", map_location="cpu")
        #state_dict = torch.load("/home/monajati/main/metaVL/magma/gpt/opt-1.3b.pt")
        model1.load_state_dict(state_dict)
        #model1.load_state_dict(torch.load("/gpt/model.pt"), map_location="cpu")
        '''
        print('gpt language model is loaded')
            
    return model
