import re

class BasicTokenizer:
    """
    Substituto focado no princípio Tabula Rasa.
    Efetua a extração sintática mapeando Unigramas locais.
    """
    def __init__(self):
        self.word2id = {"<PAD>": 0, "<UNK>": 1, "<EOS>": 2}
        self.id2word = {0: "<PAD>", 1: "<UNK>", 2: "<EOS>"}
        self.vocab_size = 3
        self.PAD_ID = 0

    def build_vocab(self, corpus: str, max_vocab: int = 10000):
        # Quebra simples por espaços e pontuação preservada
        tokens = re.findall(r'\b\w+\b|[^\w\s]', corpus.lower())
        
        # Contagem de frequências
        freqs = {}
        for t in tokens:
            freqs[t] = freqs.get(t, 0) + 1
            
        # Ordenar os mais frequentes e ignorar a cauda longa para otimização do Vector Space
        sorted_tokens = sorted(freqs.items(), key=lambda item: item[1], reverse=True)
        
        for t, _ in sorted_tokens[:max_vocab - 3]:
            if t not in self.word2id:
                self.word2id[t] = self.vocab_size
                self.id2word[self.vocab_size] = t
                self.vocab_size += 1

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        ids = [self.word2id.get(t, self.word2id["<UNK>"]) for t in tokens]
        if add_special:
            ids.append(self.word2id["<EOS>"])
        return ids

    def decode(self, ids: list[int]) -> str:
        words = []
        for i in ids:
            w = self.id2word.get(i, "")
            if w == "<EOS>":
                break
            if w != "<PAD>" and w != "<UNK>":
                words.append(w)
        return " ".join(words)

    def pad_sequence(self, ids: list[int], max_len: int) -> list[int]:
        if len(ids) > max_len:
            return ids[:max_len]
        return ids + [self.PAD_ID] * (max_len - len(ids))

# ─── Bloco de Execução Independente (Standalone) ─────────────────────────────
# Permite correr este ficheiro diretamente e testar o tokenizador em localhost
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="J.A.R.V.I.S - Tokenizer API")
    test_tokenizer = BasicTokenizer()

    class TokenizeRequest(BaseModel):
        text: str

    @app.post("/tokenize")
    async def tokenize_text(req: TokenizeRequest):
        """Rota direta para testar a extração sintática via localhost:8000"""
        # Constrói o vocabulário dinamicamente com base no input recebido para o teste
        test_tokenizer.build_vocab(req.text)
        
        # Gera a matriz de tensores
        token_ids = test_tokenizer.encode(req.text)
        
        return {
            "status": "success",
            "mensagem_original": req.text,
            "matriz_tensores": token_ids,
            "tamanho_vocabulario": test_tokenizer.vocab_size,
            "reconstrucao": test_tokenizer.decode(token_ids)
        }

    print("═"*60)
    print(" [SISTEMA] Tokenizador isolado inicializado.")
    print(" [ACESSO] API disponível em: http://localhost:8000/docs")
    print("═"*60)
    
    # Inicia o servidor local estritamente para o tokenizador
    uvicorn.run(app, host="127.0.0.1", port=8000)