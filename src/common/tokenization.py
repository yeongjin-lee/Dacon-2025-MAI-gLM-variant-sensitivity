# src/common/tokenization.py
from transformers import AutoTokenizer

def load_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

