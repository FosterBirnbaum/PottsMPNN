import os
import torch
from omegaconf import OmegaConf
import json
import argparse
import numpy as np
from tqdm import tqdm
from run_utils import process_data, score_seqs
from potts_mpnn_utils import PottsMPNN, parse_PDB
import etab_utils as etab_utils


def energy_prediction(args):

    # Load experiment configuration (OmegaConf YAML)
    cfg = OmegaConf.load(args['config'])
    dev = cfg.dev

    # Load model checkpoint and construct the PottsMPNN model for inference
    checkpoint = torch.load(cfg.model.check_path, map_location='cpu', weights_only=False) 
    model = PottsMPNN(ca_only=False, num_letters=cfg.model.vocab, vocab=cfg.model.vocab, node_features=cfg.model.hidden_dim, edge_features=cfg.model.hidden_dim, hidden_dim=cfg.model.hidden_dim, 
                            potts_dim=cfg.model.potts_dim, num_encoder_layers=cfg.model.num_layers, num_decoder_layers=cfg.model.num_layers, k_neighbors=cfg.model.num_edges, augment_eps=cfg.inference.noise)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    model = model.to(dev)
    for param in model.parameters():
        param.requires_grad = False

    # If predicting binding energies, load information about chain separation
    if cfg.inference.binding_energy_json:
        with open(cfg.inference.binding_energy_json, 'r') as f:
            binding_energy_chains = json.load(f)
    else:
        binding_energy_chains = None

    # Setup dataset settings
    pdb_list, mutant_data = process_data(cfg, binding_energy_chains)

    # Iterate over PDBs, storing predictions and per-PDB statistics
    scores_df = {'pdb': [], 'mutant': [], 'wildtype': [], 'ddG_pred': [], 'ddG_expt': []}
    stats_df = {'pdb': [], 'Pearson r': []}
    for pdb in tqdm(pdb_list):
        input_pdb = os.path.join(cfg.input_dir, pdb + '.pdb')

        # If calculating binding energy, determine which chain partitions need to be calculated in unbound state
        pdb_mutant_data = mutant_data[mutant_data['pdb'] == pdb]
        pdb_mutant_chains = []
        for mut_chains in pdb_mutant_data['mut_chains'].unique():
            pdb_mutant_chains += mut_chains.split(':')
        pdb_mutant_chains = set(pdb_mutant_chains)
        partition_flags = []
        if binding_energy_chains:
            partition_flags = [cfg.inference.ddG and len(set(partition).intersection(pdb_mutant_chains)) > 0 for partition in binding_energy_chains[pdb]]
        
        # Parse PDB
        pdb_data = parse_PDB(input_pdb, skip_gaps=cfg.inference.skip_gaps)

        # Get sequence scores for whole complex
        scores, scored_seqs, reference_scores = score_seqs(model, cfg, pdb_data, pdb_mutant_data['ddG_expt'].values, pdb_mutant_data['sequences'].values)

        # Get sequence scores for unbound state chain partitions (for binding energies only)
        p_scores = [torch.zeros_like(scores)] * len(partition_flags)
        for i_p, partition_flag in enumerate(partition_flags):
            if not partition_flag: continue
            p_scores[i_p], _, _ = score_seqs(model, cfg, pdb_data, pdb_mutant_data['ddG_expt'].values, pdb_mutant_data['partitioned_sequences'][i_p].values, binding_energy_chains[pdb][i_p])
        if len(partition_flags) > 0:
            unbound_scores = torch.cat(p_scores, dim=-1).sum(dim=-1)
            scores -= unbound_scores

        # Save info
        scores = scores.cpu().numpy()
        scored_seqs = ["".join(etab_utils.ints_to_seq_torch(seq)) for seq in scored_seqs]
        reference_scores = reference_scores.cpu().numpy()
        scores_df['pdb'] += [pdb] * len(scores)
        scores_df['mutant'] += scored_seqs
        scores_df['wildtype'] += [pdb_data['seq']] * len(scores)
        scores_df['ddG_pred'] += scores.tolist()
        scores_df['ddG_expt'] += reference_scores.tolist()
        stats_df['pdb'].append(pdb)
        stats_df['corr'].append(np.corrcoef(scores, reference_scores)[0, 1])
    
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        args_dict = json.load(f)

    energy_prediction(args_dict)