from transformers import T5EncoderModel, T5Tokenizer
import torch
import numpy as np

# Load model
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").eval()


def get_prostt5_embedding(sequence: str) -> np.ndarray:
    sequence = " ".join(list(sequence))
    ids = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        embedding = model(**ids).last_hidden_state.squeeze(0)
    return embedding.mean(dim=0).numpy() # alternatives: max, min, etc.


def aggregate_msa_embeddings(msa_sequences):
    seq_embeds = np.stack([get_prostt5_embedding(seq) for seq in msa_sequences])
    return np.concatenate([
        seq_embeds.mean(axis=0),
        seq_embeds.std(axis=0)
    ])
