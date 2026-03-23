from langchain_ollama import OllamaLLM
from langchain.schema.output_parser import StrOutputParser
import ollama


class LLMHandler:
    def __init__(self):
        self.currnet_model = None
        self.availble_models = ollama.list().models
        pass

    def get_available_models(self):
        return self.availble_models
    
    def load_model(self, model_name):
        temp_model = self.currnet_model
        for item in self.availble_models:
            if item['model'] == model_name:
                self.currnet_model = OllamaLLM(model=model_name)
        if self.currnet_model == temp_model:
            raise ValueError(f"Model {model_name} not found in local ollama list. Please select an installed model or download it.")
    
    def invoke_model(self, prompt, mappings):
        if self.currnet_model is not None:
            chain = (
                prompt
                | self.currnet_model
                | StrOutputParser()
            )
            return chain.invoke(mappings)
        else:
            raise ValueError("No model loaded. Please load a model before invoking.")