import os
import random
from collections import namedtuple
import json
from Bio import SeqIO


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
                
    # Set random seed
    random.seed(42)
    # Sample n_seq sequences
    sampled_seqs = random.sample(seqs, n_seq)
    
    return sampled_seqs


def fasta_to_json(input_fasta_files, output_dir, ntrain_seq=100, nval_seq=100, random_seed=42):
    """Write n_seq sequences from input_fasta to output_json

    Args:
        input_fasta (str): fasta file paths with protein sequences
        output_csv (str): csv file path to write sequences to
        n_seq (int, optional): Number of sequences to sample. Defaults to 100.

    Returns:
        str: output_json file path
    """


    # Convert fasta files to hierarchial dict to keep track of the sequences that correspond to the same id
    fasta_records = {}  # Nested dictionary
    for f_key, f_value in input_fasta_files.items():
        all_records = list(SeqIO.parse(f_value, "fasta"))
        # Dictionary
        for record in all_records:
            # Check if key exists
            if record.id in fasta_records:
                # Add new key
                fasta_records[record.id][f_key] = str(record.seq)
            else:
                fasta_records[record.id] = {f_key: str(record.seq)}

    # Check for each key, the length of sequence is the same as the length of catalytic_sites
    for record in fasta_records.values():
        assert len(record["sequence"]) == len(record["catalytic_sites"])
    
    # Randomly sample from fasta records
    random.seed(random_seed)
    shuffled_keys = list(fasta_records.keys())
    random.shuffle(shuffled_keys)
    
    # if ntrain_seq is "all" and nval_seq is 0, return all sequences
    if ntrain_seq == "all":
        ntrain_seq = len(shuffled_keys)
        nval_seq = 0
    elif nval_seq == "all":
        ntrain_seq = 0
        nval_seq = len(shuffled_keys)

    protein_data_train = [{"id":key, "seq":fasta_records[key]["sequence"], "catalytic_sites":fasta_records[key]["catalytic_sites"]} for key in shuffled_keys[:ntrain_seq]]
    protein_catalytic_data_train = [{"id":key, "seq":fasta_records[key]["catalytic_sites"]} for key in shuffled_keys[:ntrain_seq]]

    protein_data_val = [{"id":key, "seq":fasta_records[key]["sequence"], "catalytic_sites":fasta_records[key]["catalytic_sites"]} for key in shuffled_keys[ntrain_seq:ntrain_seq+nval_seq]]
    protein_catalytic_data_val = [{"id":key, "seq":fasta_records[key]["catalytic_sites"]} for key in shuffled_keys[ntrain_seq:ntrain_seq+nval_seq]]

    # Write to json with "seq" and "id" keys
    # Create new folders in the output dir
    seq_folder = output_dir + "/seq"
    catalytic_folder = output_dir + "/catalytic"
    if ntrain_seq != 0:
        with open(f"{output_dir}/train.json", "w") as f:
            json.dump(protein_data_train, f)
    if nval_seq != 0:
        with open(f"{output_dir}/val.json", "w") as f:
            json.dump(protein_data_val, f)
    # with open(f"{catalytic_folder}/train.json", "w") as f:
    #     json.dump(protein_catalytic_data_train, f)
    # with open(f"{catalytic_folder}/val.json", "w") as f:
    #     json.dump(protein_catalytic_data_val, f)

    return {"num_train_seq": ntrain_seq, "num_val_seq": nval_seq}

    