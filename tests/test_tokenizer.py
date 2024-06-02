import torch
from transformers import EsmForMaskedLM, EsmTokenizer, EsmConfig

from torchtitan.datasets import build_hf_data_loader
from torchtitan.models.llama.model import ModelArgs, Transformer

from plm_train.fasta_utils import sample_seqs, fasta_to_csv


def test_tokenizer():
    # Sample sequences
    protein_data = sample_seqs('data/uniprotkb_E_coli_AND_model_organism_833_2024_05_04.fasta', n_seq=10)
    model_seed = 1
    max_length = 256
    tokenizer = EsmTokenizer.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_{model_seed}')
    seq_data = [seq.seq for seq in protein_data]
    seq, attention_mask = tokenizer(seq_data, padding='max_length',
                                                  truncation=True,
                                                  max_length=max_length).values()

    assert len(seq[0]) == max_length


def test_fasta_to_csv():
    # Convert fasta to csv
    input_fasta = 'data/uniprotkb_E_coli_AND_model_organism_833_2024_05_04.fasta'
    output_csv = 'data/csv/uniprotkb_E_coli_AND_model_organism_833_2024_05_04.csv'
    
    fasta_to_csv(input_fasta, output_csv, n_seq=100)
    
    # Read csv
    with open(output_csv, 'r') as f:
        lines = f.readlines()
    
    assert len(lines) == 100


def test_dataloader():
    model_seed = 1
    tokenizer = EsmTokenizer.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_{model_seed}')
    data_loader = build_hf_data_loader('ecoli_protein', "/workspace/plm-train/data/csv", tokenizer, batch_size=8, seq_len=256, world_size=1, rank=0)

    # TODO: Add some assertions
    assert 1


def test_transformer_model():
    model_config = ModelArgs(dim=1024, n_layers=16, n_heads=8)
    model_seed = 1
    tokenizer = EsmTokenizer.from_pretrained(f'facebook/esm1v_t33_650M_UR90S_{model_seed}')
    model_config.vocab_size = len(tokenizer.all_tokens)
    model_config.max_seq_len = 256
    model = Transformer(model_config)

    assert 1
