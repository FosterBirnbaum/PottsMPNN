"""Mutation search utilities for PottsMPNN energy ranking.

This module provides a notebook-friendly wrapper that explores combinatorial
mutations by iteratively scoring all single-site variants, keeping the top
percentage at each depth, and recursing to the next mutation count.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import json
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from omegaconf import OmegaConf

from potts_mpnn_utils import PottsMPNN, parse_PDB
from run_utils import score_seqs

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class Candidate:
    sequence: str
    mutations: Tuple[str, ...]
    score: Optional[float] = None


def load_model_from_config(cfg_path: str) -> Tuple[PottsMPNN, OmegaConf]:
    """Load a PottsMPNN model and config for inference."""
    cfg = OmegaConf.load(cfg_path)
    cfg.model.vocab = 22 if "msa" in cfg.model.check_path else 21

    checkpoint = torch.load(cfg.model.check_path, map_location="cpu", weights_only=False)
    model = PottsMPNN(
        ca_only=False,
        num_letters=cfg.model.vocab,
        vocab=cfg.model.vocab,
        node_features=cfg.model.hidden_dim,
        edge_features=cfg.model.hidden_dim,
        hidden_dim=cfg.model.hidden_dim,
        potts_dim=cfg.model.potts_dim,
        num_encoder_layers=cfg.model.num_layers,
        num_decoder_layers=cfg.model.num_layers,
        k_neighbors=cfg.model.num_edges,
        augment_eps=cfg.inference.noise,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    model = model.to(cfg.dev)
    for param in model.parameters():
        param.requires_grad = False
    return model, cfg


def _parse_binding_partitions(binding_energy_json: Optional[str], pdb_name: str) -> List[List[str]]:
    if not binding_energy_json:
        return []
    with open(binding_energy_json, "r", encoding="utf-8") as handle:
        binding_data = json.load(handle)
    return binding_data.get(pdb_name, [])


def _chain_lengths(pdb_data: dict) -> Dict[str, int]:
    return {chain: len(pdb_data[f"seq_chain_{chain}"]) for chain in pdb_data["chain_order"]}


def _partition_sequence(
    sequence: str,
    chain_order: Sequence[str],
    chain_lengths: Dict[str, int],
    partition: Sequence[str],
) -> str:
    offsets = {}
    offset = 0
    for chain in chain_order:
        offsets[chain] = offset
        offset += chain_lengths[chain]
    return "".join(
        sequence[offsets[chain] : offsets[chain] + chain_lengths[chain]]
        for chain in partition
        if chain in offsets
    )


def _partition_sequences(
    sequences: Sequence[str],
    chain_order: Sequence[str],
    chain_lengths: Dict[str, int],
    partition: Sequence[str],
) -> List[str]:
    return [
        _partition_sequence(sequence, chain_order, chain_lengths, partition)
        for sequence in sequences
    ]

def _global_position_map(chain_lengths: Dict[str, int]) -> Dict[Tuple[str, int], int]:
    mapping = {}
    offset = 0
    for chain, length in chain_lengths.items():
        for pos in range(1, length + 1):
            mapping[(chain, pos)] = offset + pos - 1
        offset += length
    return mapping


def _allowed_mutations_by_position(
    chain_lengths: Dict[str, int],
    allowed_mutations: Optional[Dict[str, Dict[int, Optional[Iterable[str]]]]] = None,
    disallowed_chains: Optional[Iterable[str]] = None,
) -> Dict[int, List[str]]:
    global_positions = _global_position_map(chain_lengths)
    disallowed_chains_set = set(disallowed_chains or [])
    if not allowed_mutations:
        allowed_by_pos: Dict[int, List[str]] = {}
        offset = 0
        for chain, length in chain_lengths.items():
            if chain in disallowed_chains_set:
                offset += length
                continue
            for pos in range(length):
                allowed_by_pos[offset + pos] = AMINO_ACIDS[:]
            offset += length
        return allowed_by_pos

    allowed_by_pos: Dict[int, List[str]] = {}
    for chain, position_map in allowed_mutations.items():
        if chain not in chain_lengths:
            raise ValueError(f"Chain '{chain}' is not present in the structure.")
        if chain in disallowed_chains_set:
            continue
        if isinstance(position_map, dict):
            for pos, residues in position_map.items():
                idx = global_positions[(chain, pos)]
                if residues is None:
                    allowed_by_pos[idx] = AMINO_ACIDS[:]
                else:
                    allowed_by_pos[idx] = [res for res in residues if res in AMINO_ACIDS]
        else:
            for pos in position_map:
                idx = global_positions[(chain, pos)]
                allowed_by_pos[idx] = AMINO_ACIDS[:]
    return allowed_by_pos


def _plot_mutation_distributions(
    results: Dict[int, pd.DataFrame],
    chain_lengths: Dict[str, int],
    output_dir: Optional[str],
) -> None:
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    chain_order = list(chain_lengths.keys())

    for depth, df in results.items():
        if df.empty:
            continue
        chain_counts = {chain: np.zeros(chain_lengths[chain], dtype=int) for chain in chain_order}
        for mut_list in df["mutations"].fillna(""):
            if not mut_list:
                continue
            for mut in mut_list.split(","):
                try:
                    chain, rest = mut.split(":")
                    wt = rest[0]
                    pos = int(rest[1:-1])
                    mut_res = rest[-1]
                except ValueError:
                    continue
                _ = wt, mut_res
                if chain not in chain_counts:
                    continue
                chain_counts[chain][pos - 1] += 1

        n_chains = len(chain_order)
        fig, axes = plt.subplots(
            nrows=n_chains,
            ncols=1,
            figsize=(12, max(2.5, 2.0 * n_chains)),
            sharey=True,
        )
        if n_chains == 1:
            axes = [axes]
        for ax, chain in zip(axes, chain_order):
            counts = chain_counts[chain]
            ax.bar(np.arange(len(counts)) + 1, counts, color="#4c78a8")
            ax.set_title(f"Chain {chain}")
            ax.set_xlabel("Position")
            ax.set_xlim(0.5, len(counts) + 0.5)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        fig.suptitle(f"Mutation distribution (depth {depth})")
        axes[0].set_ylabel("Mutation count")
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        fig.savefig(os.path.join(output_dir, f"mutation_distribution_depth_{depth}.png"))
        plt.close(fig)


def _format_mutation(chain: str, pos: int, wt: str, mut: str) -> str:
    return f"{chain}:{wt}{pos}{mut}"


def _sequence_mutations(
    sequence: str,
    chain_lengths: Dict[str, int],
    allowed_by_pos: Dict[int, List[str]],
    allowed_from: Optional[Iterable[str]] = None,
    allowed_to: Optional[Iterable[str]] = None,
) -> List[Tuple[str, Tuple[str, ...]]]:
    allowed_from_set = set(allowed_from or AMINO_ACIDS)
    allowed_to_set = set(allowed_to or AMINO_ACIDS)
    chain_order = list(chain_lengths.keys())
    chain_offsets = {}
    offset = 0
    for chain in chain_order:
        chain_offsets[chain] = offset
        offset += chain_lengths[chain]

    mutants = []
    for chain in chain_order:
        start = chain_offsets[chain]
        for local_pos in range(1, chain_lengths[chain] + 1):
            global_pos = start + local_pos - 1
            if global_pos not in allowed_by_pos:
                continue
            wt = sequence[global_pos]
            if wt not in allowed_from_set:
                continue
            allowed_targets = [aa for aa in allowed_by_pos[global_pos] if aa in allowed_to_set]
            for mut in allowed_targets:
                if mut == wt:
                    continue
                new_seq = sequence[:global_pos] + mut + sequence[global_pos + 1 :]
                mutation = _format_mutation(chain, local_pos, wt, mut)
                mutants.append((new_seq, (mutation,)))
    return mutants


def _score_sequences(
    model: PottsMPNN,
    cfg: OmegaConf,
    pdb_data_list: Sequence[dict],
    sequences: Sequence[str],
    binding_partitions_list: Sequence[List[List[str]]],
    energy_mode: str,
) -> np.ndarray:
    cfg.inference.ddG = True
    cfg.inference.mean_norm = False
    cfg.inference.filter = False
    if len(pdb_data_list) != len(binding_partitions_list):
        raise ValueError("pdb_data_list and binding_partitions_list must be the same length.")

    all_scores = []
    for pdb_data, binding_partitions in zip(pdb_data_list, binding_partitions_list):
        chain_order = pdb_data[0]["chain_order"]
        chain_lengths = _chain_lengths(pdb_data[0])
        scores, _, _ = score_seqs(
            model, cfg, pdb_data, [0.0] * len(sequences), list(sequences), track_progress=True
        )
        scores = scores.squeeze(0)

        if energy_mode == "stability":
            all_scores.append(scores.cpu().numpy())
            continue

        if not binding_partitions:
            raise ValueError("Binding energy scoring requires binding_energy_json partitions.")

        unbound_scores = torch.zeros_like(scores)
        for partition in binding_partitions:
            partition_sequences = _partition_sequences(
                sequences, chain_order, chain_lengths, partition
            )
            partition_scores, _, _ = score_seqs(
                model,
                cfg,
                pdb_data,
                [0.0] * len(sequences),
                partition_sequences,
                partition=partition,
                track_progress=True,
            )
            unbound_scores = unbound_scores + partition_scores.squeeze(0)

        binding_scores = scores - unbound_scores
        if energy_mode == "binding":
            all_scores.append(binding_scores.cpu().numpy())
        elif energy_mode == "both":
            combined = scores + binding_scores
            all_scores.append(combined.cpu().numpy())
        else:
            raise ValueError("energy_mode must be one of: 'stability', 'binding', 'both'.")

    return np.mean(np.stack(all_scores, axis=0), axis=0)


def _normalize_amino_acids(amino_acids: Optional[Iterable[str]]) -> Optional[List[str]]:
    if amino_acids is None:
        return None
    normalized = [aa for aa in amino_acids if aa in AMINO_ACIDS]
    if not normalized:
        raise ValueError("Provided amino acid filter did not match any canonical residues.")
    return normalized


def recursive_mutation_search(
    pdb_paths: Union[str, Sequence[str]],
    cfg_path: str,
    max_mutations: int,
    top_percent: float,
    *,
    allowed_mutations: Optional[Dict[str, Dict[int, Optional[Iterable[str]]]]] = None,
    disallowed_chains: Optional[Iterable[str]] = None,
    binding_energy_json: Optional[str] = None,
    energy_mode: str = "stability",
    plot_dir: Optional[str] = None,
    allowed_from_aas: Optional[Iterable[str]] = None,
    allowed_to_aas: Optional[Iterable[str]] = None,
) -> Dict[int, pd.DataFrame]:
    """Search mutations iteratively and return the top percent at each depth.

    Parameters
    ----------
    pdb_paths : str or sequence of str
        Path(s) to input PDB file(s). Multiple files must share the same length.
    cfg_path : str
        Path to PottsMPNN energy prediction config (YAML).
    max_mutations : int
        Maximum number of mutations to explore.
    top_percent : float
        Percentage (0-100) of candidates to keep at each depth.
    allowed_mutations : dict, optional
        Mapping of chain -> {position (1-indexed): [allowed residues] or None}.
        If None, all positions and canonical residues are allowed.
    disallowed_chains : iterable, optional
        Chains to disallow from mutation entirely (e.g., ["B", "C"]).
    binding_energy_json : str, optional
        Path to JSON describing binding partitions for energy calculation.
    energy_mode : str
        "stability", "binding", or "both". "both" is stability + binding.
    plot_dir : str, optional
        If provided, save mutation distribution plots to this directory.
    allowed_from_aas : iterable, optional
        Amino acids that are allowed to be mutated from (wildtype filter).
    allowed_to_aas : iterable, optional
        Amino acids that are allowed to be mutated to (mutant filter).

    Returns
    -------
    dict
        Mapping of mutation count to a DataFrame with columns:
        sequence, mutations, score.
    """
    if max_mutations < 1:
        raise ValueError("max_mutations must be >= 1.")
    if not (0.0 < top_percent <= 100.0):
        raise ValueError("top_percent must be within (0, 100].")

    model, cfg = load_model_from_config(cfg_path)
    pdb_path_list = [pdb_paths] if isinstance(pdb_paths, str) else list(pdb_paths)
    if not pdb_path_list:
        raise ValueError("pdb_paths must contain at least one PDB path.")
    pdb_data_list = [parse_PDB(path, skip_gaps=cfg.inference.skip_gaps) for path in pdb_path_list]
    pdb_names = [pdb_data[0]["name"] for pdb_data in pdb_data_list]

    chain_lengths = _chain_lengths(pdb_data_list[0][0])
    total_length = sum(chain_lengths.values())
    chain_order = pdb_data_list[0][0]["chain_order"]
    for pdb_data in pdb_data_list[1:]:
        if pdb_data[0]["chain_order"] != chain_order:
            raise ValueError("All PDBs must have the same chain order.")
        if sum(_chain_lengths(pdb_data[0]).values()) != total_length:
            raise ValueError("All PDBs must have the same total sequence length.")
    allowed_by_pos = _allowed_mutations_by_position(
        chain_lengths,
        allowed_mutations,
        disallowed_chains=disallowed_chains,
    )
    binding_partitions_list = [
        _parse_binding_partitions(binding_energy_json, pdb_name) for pdb_name in pdb_names
    ]
    normalized_from = _normalize_amino_acids(allowed_from_aas)
    normalized_to = _normalize_amino_acids(allowed_to_aas)

    current = [Candidate(sequence=pdb_data_list[0][0]["seq"], mutations=tuple())]
    results: Dict[int, pd.DataFrame] = {}

    for depth in range(1, max_mutations + 1):
        generated: Dict[str, Candidate] = {}
        for candidate in current:
            for new_seq, new_mut in _sequence_mutations(
                candidate.sequence,
                chain_lengths,
                allowed_by_pos,
                allowed_from=normalized_from,
                allowed_to=normalized_to,
            ):
                mutations = candidate.mutations + new_mut
                if new_seq in generated:
                    continue
                generated[new_seq] = Candidate(sequence=new_seq, mutations=mutations)

        if not generated:
            results[depth] = pd.DataFrame(columns=["sequence", "mutations", "score"])
            current = []
            continue

        sequences = list(generated.keys())
        scores = _score_sequences(
            model, cfg, pdb_data_list, sequences, binding_partitions_list, energy_mode
        )
        for seq, score in zip(sequences, scores):
            generated[seq].score = float(score)

        ranked = sorted(generated.values(), key=lambda c: c.score)
        keep_n = max(1, ceil(len(ranked) * (top_percent / 100.0)))
        kept = ranked[:keep_n]

        results[depth] = pd.DataFrame(
            {
                "sequence": [c.sequence for c in kept],
                "mutations": [",".join(c.mutations) for c in kept],
                "score": [c.score for c in kept],
            }
        )
        current = kept

    _plot_mutation_distributions(results, chain_lengths, plot_dir)
    return results


__all__ = ["recursive_mutation_search", "load_model_from_config"]
