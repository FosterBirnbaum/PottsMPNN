from tqdm import tqdm
import os
import torch
import json
import numpy as np
import argparse
from run_utils import loss_nll, optimize_sequence, string_to_int, process_configs, alt_parse_PDB, cat_neighbors_nodes, potts_energy
import pickle
import pandas as pd
from omegaconf import OmegaConf
from potts_mpnn_utils import PottsMPNN, tied_featurize, nlcpl
import etab_utils as etab_utils

def sample_seqs(args):
    
    # Load experiment configuration (OmegaConf YAML)
    cfg = OmegaConf.load(args['config'])
    dev = cfg.platform.accel

    # Ensure output directories exist for sequences and metrics
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, "sequence_metrics"), exist_ok=True)

    # Filepaths for outputs: sampled sequences, decoding orders, per-sequence losses, and averaged losses
    filename = os.path.join(cfg.out_dir, cfg.out_name + '.fasta')
    decoding_order_filename = os.path.join(cfg.out_dir, cfg.out_name + 'decoding_order.json')
    av_loss_filename = os.path.join(cfg.out_dir, "sequence_metrics", cfg.out_name + '_av_loss.csv')

    # If sequences already exist and user wants to optimize them, set `skip_calc` so sequences can be reused.
    skip_calc = False
    if cfg.inference.potts_optimize:
        with open(filename, 'r') as f:
            seqs = f.readlines()
        seqs = {pdb.strip('>').strip(): seq.strip() for pdb, seq in zip(seqs[::2], seqs[1::2])}
        optimized_filename = os.path.join(cfg.out_dir, cfg.out_name + f'_optimized_{cfg.inference.potts_optimize}.fasta')
        print(f'Saving optimized sequences to filename {optimized_filename}.')
        skip_calc = True
    elif cfg.inference.potts_optimize:
        optimized_filename = os.path.join(cfg.out_dir, cfg.out_name + f'_optimized_{cfg.inference.potts_optimize}.fasta')
        opt_seqs = {}
        print(f'Saving sequences to filename {filename} and saving optimized sequences to filename {optimized_filename}.')
    else:
        print(f'Saving to filename {filename}.')

    # Load model checkpoint and construct the PottsMPNN model for inference
    checkpoint = torch.load(cfg.model.check_path, map_location='cpu', weights_only=False) 
    model = PottsMPNN(ca_only=False, num_letters=cfg.model.vocab, vocab=cfg.model.vocab, node_features=cfg.model.hidden_dim, edge_features=cfg.model.hidden_dim, hidden_dim=cfg.model.hidden_dim, 
                            potts_dim=cfg.model.potts_dim, num_encoder_layers=cfg.model.num_layers, num_decoder_layers=cfg.model.num_layers, k_neighbors=cfg.model.num_edges, augment_eps=cfg.inference.noise)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    model = model.to(dev)
    for param in model.parameters():
        param.requires_grad = False

    # Containers to accumulate outputs and metrics across PDBs
    out_seqs = {}
    decoding_orders = {}

    av_losses = {'pdb': [], 'seq_loss': [], 'nsr': [], 'potts_loss': []}

    # Read list of PDBs to process and derive chain mapping
    vis_chain_dicts = None
    with open(cfg.input_list, 'r') as f:
        pdb_list = f.readlines()
    pdb_list = [pdb.strip().split('_')[0] for pdb in pdb_list]
    chain_dict = {pdb: pdb.split('_')[1:] for pdb in pdb_list}

    # Load various configuration dictionaries (fixed positions, PSSMs, omissions, bias, tied positions)
    fixed_positions_dict, pssm_dict, omit_AA_dict, bias_AA_dict, tied_positions_dict, bias_by_res_dict, omit_AAs_np = process_configs(cfg)
    constant = torch.tensor(omit_AAs_np, device=dev)
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX-'
    if cfg.model.vocab == 21:
        alphabet = alphabet[:-1]  # remove gap character
    bias_AAs_np = np.zeros(len(alphabet))
    if bias_AA_dict:
            for n, AA in enumerate(alphabet):
                    if AA in list(bias_AA_dict.keys()):
                            bias_AAs_np[n] = bias_AA_dict[AA]

    # Iterate over PDBs: parse, featurize, sample sequences, compute metrics, and optionally optimize
    for pdb in tqdm(pdb_list):
        input_pdb = os.path.join(cfg.input_dir, pdb + '.pdb')
        pdb_data = alt_parse_PDB(input_pdb, chain_dict[pdb], skip_gaps=True)
        X, S_true, mask, _, chain_mask, chain_encoding_all, _, _, _, _, chain_M_pos, omit_AA_mask, residue_idx, _, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds, bias_by_res, tied_beta, chain_lens = tied_featurize([pdb_data[0]], dev, vis_chain_dicts, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict, ca_only=False, vocab=cfg.model.vocab)
        pssm_log_odds_mask = (pssm_log_odds > cfg.inference.pssm_threshold).float() #1.0 for true, 0.0 for false
        if cfg.inference.fix_decoding_order:
            torch.manual_seed(string_to_int(pdb) + cfg.inference.decoding_order_offset)
        randn = torch.randn(chain_mask.shape, device=X.device)

        # Run the encoder to obtain representations
        h_V, E_idx, h_E, etab = model.run_encoder(X, mask, residue_idx, chain_encoding_all)
        if not skip_calc:
            # Quick etab for Potts energy calculations
            etab_quick = etab_utils.merge_duplicate_pairE(etab, E_idx, denom=4)
            etab_quick = etab_quick.clone().view(etab_quick.shape[0], etab_quick.shape[1], etab_quick.shape[2], int(np.sqrt(etab_quick.shape[3])), int(np.sqrt(etab_quick.shape[3])))
            # Number of independent sampling runs and how many to keep
            num_samples = cfg.inference.num_samples

            sample_records = []
            sample_seq_loss = []
            sample_nsr = []
            sample_nlcpl = []
            # Run sampling multiple times per PDB
            for sidx in range(num_samples):
                # If decoding order is fixed, make it deterministic per-sample by adding sidx
                if cfg.inference.fix_decoding_order:
                    torch.manual_seed(string_to_int(pdb) + cfg.inference.decoding_order_offset + sidx)

                # Fresh noise for sampling; seed above ensures determinism when requested
                randn = torch.randn(chain_mask.shape, device=X.device)

                # Sample using standard or tied sampling depending on config
                if tied_positions_dict == None:
                    output_dict, all_probs = model.decoder(h_V, E_idx, h_E, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=mask, temperature=cfg.inference.temperature, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=cfg.inference.pssm_multi, pssm_log_odds_flag=bool(cfg.inference.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(cfg.inference.pssm_bias_flag), bias_by_res=bias_by_res)
                else:
                    output_dict, all_probs = model.tied_decoder(h_V, E_idx, h_E, randn, S_true, chain_mask, chain_encoding_all, residue_idx, mask=mask, temperature=cfg.inference.temperature, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=cfg.inference.pssm_multi, pssm_log_odds_flag=bool(cfg.inference.pssm_log_odds_flag), pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=bool(cfg.inference.pssm_bias_flag), tied_pos=tied_pos_list_of_lists_list[0], tied_beta=tied_beta, bias_by_res=bias_by_res)

                # Compute per-sample reporting metrics
                _, av_seq_loss, nsr = loss_nll(S_true, all_probs, chain_mask)
                sample_seq_loss.append(av_seq_loss.cpu().item())
                sample_nsr.append(nsr.cpu().item())
                _, av_nlcpl_loss = nlcpl(etab, E_idx, S_true, chain_mask)
                sample_nlcpl.append(av_nlcpl_loss.cpu().item())

                # Convert sampled integer sequence to string
                seq_str = "".join(etab_utils.ints_to_seq_torch(output_dict['S'][0]))

                # Compute full Potts energy for this sample
                seq_tensor = output_dict['S'][0].unsqueeze(0).to(dtype=torch.int64, device=E_idx.device)
                total_energy, _, _ = etab_utils.calc_eners_dense(etab_quick, E_idx, seq_tensor.unsqueeze(1), None)
                total_energy = total_energy.squeeze().cpu().item()
                sample_records.append({'sample_idx': sidx, 'seq': seq_str, 'energy': total_energy, 'decoding_order': output_dict['decoding_order']})

            # Record the samples sorted by Potts energy (lower is better)
            sample_records = sorted(sample_records, key=lambda x: x['energy'])
            for k in range(len(sample_records)):
                rec = sample_records[k]
                sidx = rec['sample_idx']
                suffix = f"_{sidx}" if sidx != 0 else ''
                out_seqs[pdb + suffix] = rec['seq']

                # Store decoding orders
                if pdb not in decoding_orders:
                    decoding_orders[pdb] = {}
                decoding_orders[pdb][suffix if suffix != '' else '_0'] = rec['decoding_order']

                # Store per-sequence losses
                av_losses['pdb'].append(pdb + suffix)
                av_losses['seq_loss'].append(sample_seq_loss[sidx])
                av_losses['nsr'].append(sample_nsr[sidx])
                av_losses['potts_loss'].append(sample_nlcpl[sidx])

        if cfg.inference.potts_optimize:
            # If requested, sequence optimization for each saved sample for this PDB
            h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
            # Iterate over saved sequences for this pdb (keys may include suffixes)
            for key in list(out_seqs.keys()):
                seq_to_opt = out_seqs[key]
                # decoding_order lookup: decoding_orders[pdb] is a dict mapping suffix->_order
                suffix_key = key[len(pdb):] if len(key) > len(pdb) else ''
                stored_decoding = None
                if pdb in decoding_orders:
                    stored_decoding = decoding_orders[pdb].get(suffix_key if suffix_key != '' else '_0', None)
                opt_seq = optimize_sequence(seq_to_opt, etab, E_idx, mask, chain_mask, cfg.inference.potts_optimize, etab_utils.seq_to_ints, model, h_E, h_EXV_encoder, h_V, constant, decoding_order=stored_decoding)
                opt_seqs[key] = etab_utils.ints_to_seq_torch(opt_seq)

    # Write optimized sequences (if any) and optionally save decoding order.
    if cfg.inference.potts_optimize:
        with open(optimized_filename, 'w') as f:
            for pdb, seq in opt_seqs.items():
                f.write('>' +  pdb + '\n' + seq + '\n')
        if not os.path.exists(decoding_order_filename):
            with open(decoding_order_filename, 'w') as f:
                json.dump(decoding_orders, f)
        if skip_calc:
            return True
        
    # Save per-sequence losses
    av_losses = pd.DataFrame(av_losses)
    av_losses.to_csv(av_loss_filename, index=None)
    
    # Write sampled sequences to FASTA
    with open(filename, 'w') as f:
        for pdb, seq in out_seqs.items():
            f.write('>' +  pdb + '\n' + seq + '\n')

    # Optionally save decoding order used during autoregressive sampling
    if args['fix_decoding_order']:
        with open(decoding_order_filename, 'w') as f:
            json.dump(decoding_orders, f)

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        args_dict = json.load(f)

    # Run sampling using configuration supplied as a JSON file
    sample_seqs(args_dict)
