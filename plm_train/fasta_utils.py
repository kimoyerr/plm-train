import random
from collections import namedtuple
import json


def sample_seqs(input_fasta, n_seq=100):
    """Sample n_seq sequences from input_fasta

    Args:
        input_fasta (str): input fasta file path
        n_seq (int, optional): Number of sequences to sample. Defaults to 100.

    Returns:
        List(str): List of sampled sequences
    """

    protein = namedtuple("protein", ["id", "seq"])
    seqs = []

    with open(input_fasta) as f:
        tmp_seq = ""
        tmp_id = ""
        for line in f:
            if line[0] == '>':
                # Save previous sequence
                if tmp_seq != "":
                    seqs.append(protein(id=tmp_id, seq=tmp_seq))
                
                # New header and empty sequence
                tmp_id = line.strip()[1:]
                tmp_seq = ""
            else:
                # Read without new line characters
                tmp_seq += line.strip()
                
    # Sample n_seq sequences
    sampled_seqs = random.sample(seqs, n_seq)
    
    return sampled_seqs


def fasta_to_json(input_fasta, output_json, n_seq=100):
    """Write n_seq sequences from input_fasta to output_json

    Args:
        input_fasta (str): fasta file path with protein sequences
        output_csv (str): csv file path to write sequences to
        n_seq (int, optional): Number of sequences to sample. Defaults to 100.

    Returns:
        str: output_cjson file path
    """

    protein_data = sample_seqs(input_fasta, n_seq=n_seq)
    # Create a dictionary with "seq" and "id" keys
    protein_data = [protein._asdict() for protein in protein_data]
    # Write to json with "seq" and "id" keys
    with open(output_json, "w") as f:
        json.dump(protein_data, f)
    
    return output_json