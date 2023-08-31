import torch
from transformers import GPTNeoForCausalLM, AutoConfig, GPT2LMHeadModel
from .utils import print_main
from pathlib import Path
from transformers.modeling_utils import no_init_weights

from transformers import AutoConfig, AutoModelForCausalLM

LANGUAGE_MODELS = [
    "gptj",
]


def gptj_config():
    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-2.7B")
    '''
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
    '''
    return config


def get_gptj(
    gradient_checkpointing: bool = True,
    from_pretrained=False,
) -> torch.nn.Module:
    """
    Loads GPTJ language model from HF
    """
    print_main("Loading GPTJ language model...")
    config = gptj_config()
    config.gradient_checkpointing = gradient_checkpointing
    if gradient_checkpointing:
        config.use_cache = False
    config.model_device = "cpu"
    if from_pretrained:
        raise NotImplemented("GPTJ pretrained not implemented")
    else:
        #with no_init_weights():
        
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
        
        model1 = GPTNeoForCausalLM(config=config)

        torch.save(model.state_dict(),"/home/monajati/main/metaVL/magma/gpt/gptneo-2.7B.pt")
        #model.save_pretrained(save_directory = "/gpt/")
        #model1 = AutoModelForCausalLM.from_config(config)
        #model = AutoModelForCausalLM.from_pretrained(gpt2, state_dict=state_dict)
        state_dict = torch.load("/home/monajati/main/metaVL/magma/gpt/gptneo-2.7B.pt", map_location="cpu")
        model1.load_state_dict(state_dict)
        #model1.load_state_dict(torch.load("/gpt/model.pt"), map_location="cpu")
        #model = MetaICLModel()
        #model.load("/workspace/magma/magma/meta/MetaICL/checkpoints/channel-metaicl/qa_to_qa/model.pt")
        #model = GPTNeoForCausalLM(config=config)
        
        print('gpt language model is loaded')
        print('done')
    return model1