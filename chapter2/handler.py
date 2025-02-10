from typing import Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class WeightedAdapterHandler:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        # Base model and adapter IDs
        self.base_model_id = "meta-llama/Llama-3.1-70B"
        self.domain_adapter_id = "mwatkins1970/llama-domain"
        self.voice_adapters = {
            "davinci": "mwatkins1970/llama-davinci",
            "davinci-instruct": "mwatkins1970/llama-davinci-instruct",
            "text-davinci": "mwatkins1970/llama-text-davinci",
            "opus-calm": "mwatkins1970/llama-opus-calm",
            "opus-manic": "mwatkins1970/llama-opus-manic"
        }
    
    def load_model(self):
        """Load the base model and apply domain adapter"""
        print("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        
        print("Applying domain adapter...")
        self.model = PeftModel.from_pretrained(
            self.model,
            self.domain_adapter_id
        )
    
    def apply_voice_adapters(self, weights: Dict[str, float]):
        """Apply voice adapters with specified weights"""
        for voice_name, weight in weights.items():
            if weight != 0 and voice_name in self.voice_adapters:
                adapter_id = self.voice_adapters[voice_name]
                print(f"Applying {voice_name} adapter with weight {weight}...")
                
                voice_adapter = PeftModel.from_pretrained(
                    self.model,
                    adapter_id
                )
                
                # Merge weighted adapter parameters
                for name, param in voice_adapter.named_parameters():
                    if "lora" in name.lower():
                        base_param = self.model.get_parameter(name)
                        with torch.no_grad():
                            base_param.data += param.data * weight
    
    def __call__(self, inputs: Dict):
        """Handle inference request"""
        # Primary fields
        prompt = inputs.get("inputs", "")
        voice_weights = inputs.get("voice_weights", {
            "davinci": 1.0,  # default
            "davinci-instruct": 0.0,
            "text-davinci": 0.0,
            "opus-calm": 0.0,
            "opus-manic": 0.0
        })
        
        # If the caller provided a "parameters" block, read from there; else fall back to top-level.
        parameters = inputs.get("parameters", {})
        
        # "max_length" or "max_new_tokens"
        #  - If present in 'parameters', it's 'max_new_tokens'
        #  - Otherwise fallback to top-level 'max_length'
        max_length = parameters.get("max_new_tokens")
        if max_length is None:
            max_length = inputs.get("max_length", 100)
        
        # Temperature
        temperature = parameters.get("temperature", inputs.get("temperature", 0.7))
        
        # top_p
        top_p = parameters.get("top_p", inputs.get("top_p", 0.9))
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
            self.apply_voice_adapters(voice_weights)
            # Merge and unload means we physically bake in the LoRAs so we can drop them from memory
            self.model = self.model.merge_and_unload()
        
        # Tokenize
        tokenized = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        outputs = self.model.generate(
            **tokenized,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
        
        # Decode
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"generated_text": response_text}

# Initialize handler
handler = WeightedAdapterHandler()
