# Arquivo: Arquivos/engine_AI.py

import torch
import torch_directml

# IMPORTAÇÃO CORRIGIDA: Aponta diretamente para o seu arquivo model.py local
# e puxa as classes reais que construímos para o seu sistema.
from model import JarvisTransformer, JarvisConfig

class JarvisEngine:
    def __init__(self):
        # 1. Identifica a APU Vega 7 via DirectML
        self.device = torch_directml.device()
        print(f"Hardware alocado: {self.device}")

        # 2. Carrega as configurações (context_len, embed_dim, etc.)
        self.config = JarvisConfig(
            vocab_size=50000, # Ajuste conforme o seu tokenizer
            context_len=4096,
            embed_dim=512,
            num_heads=256,
            num_layers=256
        )

        # 3. Instancia o modelo e move para a GPU
        # Nota: O .to() é aplicado ao modelo PyTorch, não à configuração.
        self.model = JarvisTransformer(self.config).to(self.device)

        # Opcional: Carregar pesos pré-treinados
        # self.model.load_state_dict(torch.load("pesos.pth", map_location=self.device))

        self.model.eval() # Modo de inferência

    def generate_response(self, input_tensor):
        # Move os tensores de entrada para a Vega 7 antes da inferência
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        return output