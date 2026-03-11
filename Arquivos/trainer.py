import torch
import torch.nn.functional as F
from typing import Callable, Dict

class AITrainer:
    """
    Subrotina Causal Language Modeling. 
    Mede a Perda (Cross-Entropy Loss) e calcula as derivadas parciais.
    """
    def __init__(self, model: torch.nn.Module, tokenizer, corpus: str, batch_size: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4)
        
        # Tokeniza o corpus inteiro
        raw_ids = self.tokenizer.encode(corpus)
        self.data_tensor = torch.tensor(raw_ids, dtype=torch.long)

    def get_batch(self, context_len: int):
        """Gera blocos aleatórios X e Y (Y é o alvo deslocado em +1 no tempo)."""
        max_idx = len(self.data_tensor) - context_len - 1
        if max_idx <= 0:
            # Fallback se o documento for extremamente curto
            x = self.data_tensor[:-1].unsqueeze(0)
            y = self.data_tensor[1:].unsqueeze(0)
            return x.to(self.device), y.to(self.device)
            
        ix = torch.randint(0, max_idx, (self.batch_size,))
        x = torch.stack([self.data_tensor[i:i+context_len] for i in ix])
        y = torch.stack([self.data_tensor[i+1:i+context_len+1] for i in ix])
        return x.to(self.device), y.to(self.device)

    def train_loop(self, epochs: int, callback: Callable[[Dict], None]):
        self.model.train()
        steps_per_epoch = 20 # Número arbitrário de passagens matriciais para simular treino rápido
        
        for epoch in range(1, epochs + 1):
            total_loss = 0
            for _ in range(steps_per_epoch):
                X, Y = self.get_batch(self.model.cfg.context_len)
                
                # Forward Pass
                logits, loss = self.model(X, targets=Y)
                
                # Backward Pass (Retropropagação)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                
                # Atualização dos Pesos (AdamW)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / steps_per_epoch
            callback({
                "epoch": epoch,
                "total": epochs,
                "percent": int((epoch / epochs) * 100),
                "loss": avg_loss
            })