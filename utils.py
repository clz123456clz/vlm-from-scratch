from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)
    
    print("✅ Loaded config keys:", list(model_config_file.keys()))
    print("✅ PaliGemmaConfig fields:", vars(config))
    print("✅ Vision config:", config.vision_config)
    print("✅ Text config:", config.text_config)

    print(type(config.vision_config))
    print(type(config.text_config))

    assert config.vocab_size == 257216
    assert config.image_token_index == 257152
    assert config.vision_config.image_size == 224
    assert config.vision_config.hidden_size == 1152
    assert config.text_config.hidden_size == 2048

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)
    print("📦 Model configuration summary:")
    print(model.config)
    print("Vision embed dim:", model.config.vision_config.hidden_size)
    print("Text hidden size:", model.config.text_config.hidden_size)
    print("Vocab size:", model.config.vocab_size)
    print("✅ Weight count:", sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    print("=== CONFIG SANITY CHECK ===")
    print("vocab_size:", config.vocab_size)
    print("projection_dim:", config.projection_dim)
    print("image_token_index:", config.image_token_index)
    print("vision hidden:", config.vision_config.hidden_size)
    print("vision image_size:", config.vision_config.image_size)
    print("text hidden:", config.text_config.hidden_size)
    print("============================")       

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)