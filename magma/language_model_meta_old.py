import torch
from transformers import GPTNeoForCausalLM, AutoConfig, GPT2LMHeadModel
from .utils import print_main
from pathlib import Path
from transformers.modeling_utils import no_init_weights

from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

from transformers import AutoConfig, AutoModelForCausalLM


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

'''

def gpt2_config():
    config = AutoConfig.from_pretrained("gpt2-medium")
    config.attention_layers = ["global"] * 28
    config.attention_types = [["global"], 28]
    config.n_layers = 28
    config.n_heads = 16
    config.n_embd = 256 * config.n_heads
    config.vocab_size = 50400
    config.rotary = True
    config.rotary_dim = 64
    config.jax = True
    config.gradient_checkpointing = True
    return config

def gptj_config():
    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
    config.attention_layers = ["global"] * 28
    config.attention_types = [["global"], 28]
    config.num_layers = 28
    config.num_heads = 16
    config.hidden_size = 256 * config.num_heads
    config.vocab_size = 50400
    config.rotary = True
    config.rotary_dim = 64
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
    #config = gptj_config()
    config = AutoConfig.from_pretrained("gpt2-medium")
    #print(config)
    #print(begh)
    
    config.gradient_checkpointing = True
    
    #print("config",config)
    
    #config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
    #print(config)
    #print(begh)
    #config.gradient_checkpointing = gradient_checkpointing
    if gradient_checkpointing:
        config.use_cache = False
    config.model_device = "cpu"
    if from_pretrained:
        raise NotImplemented("GPTJ pretrained not implemented")
    else:
        #with no_init_weights():
        print('loading...')
        #model = AutoModelForCausalLM.from_config(config)
        #model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

        #torch.save(model.state_dict(),"/home/monajati/main/metaVL/magma/gpt/gpt2-medium.pt")
        #model.save_pretrained(save_directory = "/gpt/")
        model1 = AutoModelForCausalLM.from_config(config)
        #model = AutoModelForCausalLM.from_pretrained(gpt2, state_dict=state_dict)
        state_dict = torch.load("/home/monajati/main/metaVL/magma/gpt/model-80000.pt", map_location="cpu")
        
        model1.load_state_dict(state_dict)
        #model1.load_state_dict(torch.load("/gpt/model.pt"), map_location="cpu")
        #model = MetaICLModel()
        #model.load("/workspace/magma/magma/meta/MetaICL/checkpoints/channel-metaicl/qa_to_qa/model.pt")
        #model = GPTNeoForCausalLM(config=config)
        print('gpt language model is loaded')
    return model1


