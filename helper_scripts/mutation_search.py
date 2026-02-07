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
from run_utils import chain_to_partition_map, inter_partition_contact_mask, score_seqs

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class Candidate:
    sequence: str
    mutations: Tuple[str, ...]
    positions: Tuple[int, ...]
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


def _concat_ca_positions(pdb_entry: dict) -> torch.Tensor:
    coords = []
    for chain in pdb_entry["chain_order"]:
        chain_coords = pdb_entry[f"coords_chain_{chain}"][f"CA_chain_{chain}"]
        coords.append(np.array(chain_coords, dtype=np.float32))
    ca_pos = np.concatenate(coords, axis=0)
    return torch.from_numpy(ca_pos).unsqueeze(0)


def _chain_encoding(chain_lengths: Dict[str, int], chain_order: Sequence[str]) -> torch.Tensor:
    encoding = []
    for idx, chain in enumerate(chain_order, start=1):
        encoding.extend([idx] * chain_lengths[chain])
    return torch.tensor([encoding], dtype=torch.long)


def _interface_mask(
    pdb_entry: dict,
    chain_lengths: Dict[str, int],
    binding_partitions: List[List[str]],
    binding_energy_cutoff: float,
) -> np.ndarray:
    ca_pos = _concat_ca_positions(pdb_entry)
    chain_order = pdb_entry["chain_order"]
    chain_encoding_all = _chain_encoding(chain_lengths, chain_order).to(device=ca_pos.device)
    partition_index = chain_to_partition_map(chain_encoding_all, chain_order, binding_partitions)
    inter_mask = inter_partition_contact_mask(ca_pos, partition_index, binding_energy_cutoff)
    return inter_mask.squeeze(0).cpu().numpy().astype(bool)


def _mask_sequence_to_interface(
    sequence: str,
    wt_sequence: str,
    interface_mask: np.ndarray,
) -> str:
    return "".join(
        seq_res if interface_mask[idx] else wt_sequence[idx]
        for idx, seq_res in enumerate(sequence)
    )


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


def _parse_mutation_list(mutation_str: str) -> List[str]:
    if not mutation_str:
        return []
    return [mutation for mutation in mutation_str.split(",") if mutation]


def _aggregate_mutation_stats(
    results: Dict[int, pd.DataFrame],
    value_column: Optional[str],
) -> Tuple[Dict[int, Dict[str, int]], Dict[int, Dict[str, float]]]:
    counts_by_depth: Dict[int, Dict[str, int]] = {}
    values_by_depth: Dict[int, Dict[str, float]] = {}

    for depth, df in results.items():
        counts: Dict[str, int] = {}
        value_sums: Dict[str, float] = {}
        value_counts: Dict[str, int] = {}
        for _, row in df.iterrows():
            mutations = _parse_mutation_list(row.get("mutations", ""))
            if not mutations:
                continue
            value = None
            if value_column and value_column in row and pd.notna(row[value_column]):
                value = float(row[value_column])
            for mutation in mutations:
                counts[mutation] = counts.get(mutation, 0) + 1
                if value is not None:
                    value_sums[mutation] = value_sums.get(mutation, 0.0) + value
                    value_counts[mutation] = value_counts.get(mutation, 0) + 1
        counts_by_depth[depth] = counts
        values_by_depth[depth] = {
            mutation: value_sums[mutation] / value_counts[mutation]
            for mutation in value_sums
            if value_counts[mutation] > 0
        }
    return counts_by_depth, values_by_depth


def _plot_lineage_alluvial(
    results: Dict[int, pd.DataFrame],
    output_dir: Optional[str],
    top_n: int = 20,
) -> None:
    if not output_dir:
        return
    if not results:
        return
    os.makedirs(output_dir, exist_ok=True)

    max_depth = max(results.keys())
    if max_depth < 2:
        return

    score_column = "score" if {"stability_score", "binding_score"}.issubset(
        results[max_depth].columns
    ) else None

    step_counts: Dict[int, Dict[str, int]] = {}
    step_scores: Dict[int, Dict[str, List[float]]] = {}
    transition_counts: Dict[Tuple[int, str, str], int] = {}

    for depth, df in results.items():
        for _, row in df.iterrows():
            mutations = _parse_mutation_list(row.get("mutations", ""))
            if len(mutations) < 2:
                continue
            score = None
            if score_column and score_column in row and pd.notna(row[score_column]):
                score = float(row[score_column])
            for step, mutation in enumerate(mutations, start=1):
                step_counts.setdefault(step, {})
                step_scores.setdefault(step, {})
                step_counts[step][mutation] = step_counts[step].get(mutation, 0) + 1
                if score is not None:
                    step_scores[step].setdefault(mutation, []).append(score)
                for step in range(1, len(mutations)):
                    transition_counts[(step, mutations[step - 1], mutations[step])] = (
                        transition_counts.get((step, mutations[step - 1], mutations[step]), 0) + 1
                    )

    step_nodes: Dict[int, List[str]] = {}
    for step in range(1, max_depth + 1):
        counts = step_counts.get(step, {})
        top_mutations = sorted(counts, key=counts.get, reverse=True)[:top_n]
        step_nodes[step] = top_mutations

    fig, ax = plt.subplots(figsize=(1.8 * max_depth, 5.8))
    ax.axis("off")

    max_total = max(sum(step_counts.get(step, {}).get(m, 0) for m in step_nodes.get(step, []))
                    for step in range(1, max_depth + 1))
    if max_total == 0:
        plt.close(fig)
        return

    score_min, score_max = 0.0, 1.0
    if score_column:
        all_scores = [
            score
            for scores in step_scores.values()
            for score_list in scores.values()
            for score in score_list
        ]
        if all_scores:
            score_min = min(all_scores)
            score_max = max(all_scores)

    node_positions: Dict[Tuple[int, str], Tuple[float, float, float]] = {}
    for step in range(1, max_depth + 1):
        counts = step_counts.get(step, {})
        nodes = step_nodes.get(step, [])
        total = sum(counts.get(node, 0) for node in nodes)
        if total == 0:
            continue
        y = 0.0
        for node in nodes:
            height = counts.get(node, 0) / max_total
            node_positions[(step, node)] = (step, y, height)
            y += height + 0.02

    cmap = plt.get_cmap("coolwarm")
    for (step, node), (x, y, height) in node_positions.items():
        avg_score = None
        if score_column:
            scores = step_scores.get(step, {}).get(node, [])
            if scores:
                avg_score = sum(scores) / len(scores)
        color = cmap(0.5)
        if avg_score is not None:
            if score_max == score_min:
                color = cmap(0.5)
            else:
                color = cmap((avg_score - score_min) / (score_max - score_min))
        ax.add_patch(plt.Rectangle((x - 0.14, y), 0.28, height, color=color, alpha=0.85))
        ax.text(x, y + height / 2, node, ha="center", va="center", fontsize=8)

    for (step, source, target), count in transition_counts.items():
        if source not in step_nodes.get(step, []) or target not in step_nodes.get(step + 1, []):
            continue
        src = node_positions.get((step, source))
        tgt = node_positions.get((step + 1, target))
        if not src or not tgt:
            continue
        _, y1, h1 = src
        _, y2, h2 = tgt
        ax.plot(
            [step + 0.12, step + 0.88],
            [y1 + h1 / 2, y2 + h2 / 2],
            color="#888888",
            alpha=min(0.8, 0.3 + 0.5 * count / max_total),
            linewidth=0.9,
        )

    fig.suptitle("Mutation lineage (alluvial)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mutation_lineage_alluvial.png"))
    plt.close(fig)


def _plot_mutation_persistence_heatmap(
    results: Dict[int, pd.DataFrame],
    output_dir: Optional[str],
    top_n: int = 50,
) -> None:
    if not output_dir or not results:
        return
    os.makedirs(output_dir, exist_ok=True)

    max_depth = max(results.keys())
    score_column = "score" if {"stability_score", "binding_score"}.issubset(
        results[max_depth].columns
    ) else None

    counts_by_depth, values_by_depth = _aggregate_mutation_stats(results, score_column)
    total_counts: Dict[str, int] = {}
    for counts in counts_by_depth.values():
        for mutation, count in counts.items():
            total_counts[mutation] = total_counts.get(mutation, 0) + count

    top_mutations = sorted(total_counts, key=total_counts.get, reverse=True)[:top_n]
    if not top_mutations:
        return

    data = np.zeros((len(top_mutations), max_depth))
    for col, depth in enumerate(range(1, max_depth + 1)):
        for row, mutation in enumerate(top_mutations):
            if score_column:
                data[row, col] = values_by_depth.get(depth, {}).get(mutation, np.nan)
            else:
                data[row, col] = counts_by_depth.get(depth, {}).get(mutation, 0)

    fig, ax = plt.subplots(figsize=(1.0 * max_depth + 1.5, 0.32 * len(top_mutations) + 2))
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_yticks(np.arange(len(top_mutations)))
    ax.set_yticklabels(top_mutations, fontsize=7)
    ax.set_xticks(np.arange(max_depth))
    ax.set_xticklabels([str(depth) for depth in range(1, max_depth + 1)])
    ax.set_xlabel("Depth")
    ax.set_title(
        "Mutation persistence heatmap"
        + (" (mean score)" if score_column else " (frequency)")
    )
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mutation_persistence_heatmap.png"))
    plt.close(fig)


def _plot_mutation_cooccurrence_networks(
    results: Dict[int, pd.DataFrame],
    output_dir: Optional[str],
    top_n: int = 25,
    max_edges: int = 200,
) -> None:
    if not output_dir or not results:
        return
    os.makedirs(output_dir, exist_ok=True)

    for depth, df in results.items():
        mutation_counts: Dict[str, int] = {}
        mutation_scores: Dict[str, List[float]] = {}
        cooccurrence_counts: Dict[Tuple[str, str], int] = {}
        score_column = "score" if {"stability_score", "binding_score"}.issubset(
            df.columns
        ) else None

        for _, row in df.iterrows():
            mutations = _parse_mutation_list(row.get("mutations", ""))
            if len(mutations) < 2:
                continue
            score = None
            if score_column and score_column in row and pd.notna(row[score_column]):
                score = float(row[score_column])
            for mutation in mutations:
                mutation_counts[mutation] = mutation_counts.get(mutation, 0) + 1
                if score is not None:
                    mutation_scores.setdefault(mutation, []).append(score)
            for i in range(len(mutations)):
                for j in range(i + 1, len(mutations)):
                    pair = tuple(sorted((mutations[i], mutations[j])))
                    cooccurrence_counts[pair] = cooccurrence_counts.get(pair, 0) + 1

        top_mutations = sorted(mutation_counts, key=mutation_counts.get, reverse=True)[:top_n]
        if len(top_mutations) < 2:
            continue
        positions = {}
        angle_step = 2 * np.pi / len(top_mutations)
        for idx, mutation in enumerate(top_mutations):
            angle = idx * angle_step
            positions[mutation] = (np.cos(angle), np.sin(angle))

        edges = [
            (pair, count)
            for pair, count in cooccurrence_counts.items()
            if pair[0] in top_mutations and pair[1] in top_mutations
        ]
        edges = sorted(edges, key=lambda item: item[1], reverse=True)[:max_edges]

        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        ax.axis("off")

        max_count = max(mutation_counts[m] for m in top_mutations)
        cmap = plt.get_cmap("coolwarm")
        score_min, score_max = 0.0, 1.0
        if score_column:
            all_scores = [score for scores in mutation_scores.values() for score in scores]
            if all_scores:
                score_min = min(all_scores)
                score_max = max(all_scores)

        for (m1, m2), count in edges:
            x1, y1 = positions[m1]
            x2, y2 = positions[m2]
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="#888888",
                linewidth=0.5 + 2.0 * count / max_count,
                alpha=0.55,
            )

        for mutation in top_mutations:
            x, y = positions[mutation]
            size = 180 + 520 * mutation_counts[mutation] / max_count
            color = "#4c78a8"
            if score_column:
                scores = mutation_scores.get(mutation, [])
                if scores:
                    avg_score = sum(scores) / len(scores)
                    if score_max == score_min:
                        color = cmap(0.5)
                    else:
                        color = cmap((avg_score - score_min) / (score_max - score_min))
            ax.scatter([x], [y], s=size, color=color, edgecolor="black", linewidth=0.5)
            ax.text(x, y, mutation, ha="center", va="center", fontsize=7)

        ax.set_title(f"Mutation co-occurrence (depth {depth})")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"mutation_cooccurrence_depth_{depth}.png"))
        plt.close(fig)


def _plot_pareto_fronts(
    results: Dict[int, pd.DataFrame],
    output_dir: Optional[str],
) -> None:
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)

    for depth, df in results.items():
        if df.empty or "pareto_front" not in df.columns:
            continue
        if not {"stability_score", "binding_score"}.issubset(df.columns):
            continue

        pareto_mask = df["pareto_front"].fillna(False).to_numpy(dtype=bool)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.scatter(
            df["stability_score"],
            df["binding_score"],
            color="#4c78a8",
            alpha=0.6,
            label="Candidates",
        )
        if pareto_mask.any():
            ax.scatter(
                df.loc[pareto_mask, "stability_score"],
                df.loc[pareto_mask, "binding_score"],
                color="#f58518",
                edgecolor="black",
                linewidth=0.6,
                label="Pareto front",
            )
        ax.set_title(f"Pareto front (depth {depth})")
        ax.set_xlabel("Stability score")
        ax.set_ylabel("Binding score")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"pareto_front_depth_{depth}.png"))
        plt.close(fig)


def _format_mutation(chain: str, pos: int, wt: str, mut: str) -> str:
    return f"{chain}:{wt}{pos}{mut}"


def _sequence_mutations(
    sequence: str,
    chain_lengths: Dict[str, int],
    allowed_by_pos: Dict[int, List[str]],
    allowed_from: Optional[Iterable[str]] = None,
    allowed_to: Optional[Iterable[str]] = None,
    disallow_positions: Optional[Iterable[int]] = None,
) -> List[Tuple[str, Tuple[str, ...], int]]:
    allowed_from_set = set(allowed_from or AMINO_ACIDS)
    allowed_to_set = set(allowed_to or AMINO_ACIDS)
    disallow_positions_set = set(disallow_positions or [])
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
            if global_pos in disallow_positions_set:
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
                mutants.append((new_seq, (mutation,), global_pos))
    return mutants


def _score_sequences(
    model: PottsMPNN,
    cfg: OmegaConf,
    pdb_data_list: Sequence[dict],
    sequences: Sequence[str],
    binding_partitions_list: Sequence[List[List[str]]],
    energy_mode: str,
    binding_energy_cutoff: Optional[float] = None,
    rrf_k: int = 60,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    cfg.inference.ddG = True
    cfg.inference.mean_norm = False
    cfg.inference.filter = False
    if len(pdb_data_list) != len(binding_partitions_list):
        raise ValueError("pdb_data_list and binding_partitions_list must be the same length.")

    all_scores = []
    all_stability = []
    all_binding = []
    for pdb_data, binding_partitions in zip(pdb_data_list, binding_partitions_list):
        pdb_entry = pdb_data[0]
        chain_order = pdb_entry["chain_order"]
        chain_lengths = _chain_lengths(pdb_entry)
        wt_sequence = pdb_entry["seq"]
        scores, _, _ = score_seqs(
            model, cfg, pdb_data, [0.0] * len(sequences), list(sequences), track_progress=True
        )
        scores = scores.squeeze(0)

        stability_scores = scores.cpu().numpy()
        if energy_mode == "stability":
            all_scores.append(stability_scores)
            all_stability.append(stability_scores)
            continue

        if not binding_partitions:
            raise ValueError("Binding energy scoring requires binding_energy_json partitions.")

        interface_mask = None
        if binding_energy_cutoff is not None:
            interface_mask = _interface_mask(
                pdb_entry, chain_lengths, binding_partitions, binding_energy_cutoff
            )
            binding_sequences = [
                _mask_sequence_to_interface(seq, wt_sequence, interface_mask)
                for seq in sequences
            ]
        else:
            binding_sequences = list(sequences)

        bound_scores = torch.zeros_like(scores)
        bound_indices = [idx for idx, seq in enumerate(binding_sequences) if seq != wt_sequence]
        if bound_indices:
            bound_subset = [binding_sequences[idx] for idx in bound_indices]
            bound_subset_scores, _, _ = score_seqs(
                model,
                cfg,
                pdb_data,
                [0.0] * len(bound_subset),
                bound_subset,
                track_progress=True,
            )
            bound_scores[bound_indices] = bound_subset_scores.squeeze(0)

        unbound_scores = torch.zeros_like(scores)
        for partition in binding_partitions:
            wt_partition_seq = _partition_sequence(
                wt_sequence, chain_order, chain_lengths, partition
            )
            partition_sequences = _partition_sequences(
                binding_sequences, chain_order, chain_lengths, partition
            )
            partition_indices = [
                idx
                for idx, seq in enumerate(partition_sequences)
                if seq != wt_partition_seq
            ]
            if not partition_indices:
                continue
            partition_subset = [partition_sequences[idx] for idx in partition_indices]
            partition_scores, _, _ = score_seqs(
                model,
                cfg,
                pdb_data,
                [0.0] * len(partition_subset),
                partition_subset,
                partition=partition,
                track_progress=True,
            )
            unbound_scores[partition_indices] = (
                unbound_scores[partition_indices] + partition_scores.squeeze(0)
            )

        binding_scores = bound_scores - unbound_scores
        binding_scores_np = binding_scores.cpu().numpy()
        if energy_mode == "binding":
            all_scores.append(binding_scores_np)
            all_binding.append(binding_scores_np)
        elif energy_mode == "both":
            stability_ranks = _rank_scores(stability_scores)
            binding_ranks = _rank_scores(binding_scores_np)
            rrf_scores = _rrf_scores(stability_ranks, binding_ranks, rrf_k)
            all_scores.append(-rrf_scores)
            all_stability.append(stability_scores)
            all_binding.append(binding_scores_np)
        else:
            raise ValueError("energy_mode must be one of: 'stability', 'binding', 'both'.")

    return (
        np.mean(np.stack(all_scores, axis=0), axis=0),
        np.mean(np.stack(all_stability, axis=0), axis=0) if all_stability else None,
        np.mean(np.stack(all_binding, axis=0), axis=0) if all_binding else None,
    )


def _rank_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def _rrf_scores(stability_ranks: np.ndarray, binding_ranks: np.ndarray, rrf_k: int) -> np.ndarray:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer.")
    return (1.0 / (rrf_k + stability_ranks)) + (1.0 / (rrf_k + binding_ranks))


def _pareto_front(stability_scores: np.ndarray, binding_scores: np.ndarray) -> np.ndarray:
    n = len(stability_scores)
    front = np.ones(n, dtype=bool)
    for i in range(n):
        if not front[i]:
            continue
        for j in range(n):
            if i == j or not front[i]:
                continue
            if (
                stability_scores[j] <= stability_scores[i]
                and binding_scores[j] <= binding_scores[i]
                and (
                    stability_scores[j] < stability_scores[i]
                    or binding_scores[j] < binding_scores[i]
                )
            ):
                front[i] = False
                break
    return front


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
    binding_energy_cutoff: Optional[float] = None,
    energy_mode: str = "stability",
    rrf_k: int = 60,
    show_pareto_front: bool = False,
    plot_dir: Optional[str] = None,
    lineage_top_n: int = 20,
    heatmap_top_n: int = 50,
    cooccurrence_top_n: int = 25,
    cooccurrence_max_edges: int = 200,
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
    binding_energy_cutoff : float, optional
        Cα distance cutoff (Angstroms) for interface residues used in binding energy.
    energy_mode : str
        "stability", "binding", or "both". "both" is stability + binding.
    rrf_k : int
        Reciprocal rank fusion constant used when energy_mode is "both".
    show_pareto_front : bool
        If True and energy_mode is "both", include a Pareto front indicator column.
    plot_dir : str, optional
        If provided, save mutation distribution plots to this directory.
    lineage_top_n : int
        Top-N mutations to include per step in the alluvial lineage plot.
    heatmap_top_n : int
        Top-N mutations to include in the persistence heatmap.
    cooccurrence_top_n : int
        Top-N mutations to include in co-occurrence network plots.
    cooccurrence_max_edges : int
        Maximum number of edges to draw in co-occurrence network plots.
    allowed_from_aas : iterable, optional
        Amino acids that are allowed to be mutated from (wildtype filter).
    allowed_to_aas : iterable, optional
        Amino acids that are allowed to be mutated to (mutant filter).

    Returns
    -------
    dict
        Mapping of mutation count to a DataFrame with columns:
        sequence, mutations, score.
        Mutations at each depth are enforced to occur at distinct positions.
    """
    if max_mutations < 1:
        raise ValueError("max_mutations must be >= 1.")
    if not (0.0 < top_percent <= 100.0):
        raise ValueError("top_percent must be within (0, 100].")
    if binding_energy_cutoff is not None and binding_energy_cutoff <= 0:
        raise ValueError("binding_energy_cutoff must be a positive distance in Angstroms.")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be a positive integer.")
    if show_pareto_front and energy_mode != "both":
        raise ValueError("show_pareto_front requires energy_mode='both'.")

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
    if binding_energy_cutoff is not None and energy_mode != "stability":
        if not binding_partitions_list or not binding_partitions_list[0]:
            raise ValueError("Binding energy cutoff requires binding_energy_json partitions.")
        interface_mask = _interface_mask(
            pdb_data_list[0][0],
            chain_lengths,
            binding_partitions_list[0],
            binding_energy_cutoff,
        )
        allowed_by_pos = {
            pos: residues for pos, residues in allowed_by_pos.items() if interface_mask[pos]
        }
    normalized_from = _normalize_amino_acids(allowed_from_aas)
    normalized_to = _normalize_amino_acids(allowed_to_aas)

    current = [Candidate(sequence=pdb_data_list[0][0]["seq"], mutations=tuple(), positions=tuple())]
    results: Dict[int, pd.DataFrame] = {}

    for depth in range(1, max_mutations + 1):
        print(f"Scoring mutations at depth {depth}")
        generated: Dict[str, Candidate] = {}
        for candidate in current:
            for new_seq, new_mut, global_pos in _sequence_mutations(
                candidate.sequence,
                chain_lengths,
                allowed_by_pos,
                allowed_from=normalized_from,
                allowed_to=normalized_to,
                disallow_positions=candidate.positions,
            ):
                mutations = candidate.mutations + new_mut
                if new_seq in generated:
                    continue
                generated[new_seq] = Candidate(
                    sequence=new_seq,
                    mutations=mutations,
                    positions=candidate.positions + (global_pos,),
                )

        if not generated:
            results[depth] = pd.DataFrame(columns=["sequence", "mutations", "score"])
            current = []
            continue

        sequences = list(generated.keys())
        print(f"Scoring {len(sequences)} mutations.")
        scores, stability_scores, binding_scores = _score_sequences(
            model,
            cfg,
            pdb_data_list,
            sequences,
            binding_partitions_list,
            energy_mode,
            binding_energy_cutoff=binding_energy_cutoff,
            rrf_k=rrf_k,
        )
        for seq, score in zip(sequences, scores):
            generated[seq].score = float(score)

        ranked = sorted(generated.values(), key=lambda c: c.score)
        keep_n = max(1, ceil(len(ranked) * (top_percent / 100.0)))
        kept = ranked[:keep_n]

        data = {
            "sequence": [c.sequence for c in kept],
            "mutations": [",".join(c.mutations) for c in kept],
            "score": [c.score for c in kept],
        }
        if energy_mode == "both":
            if stability_scores is None or binding_scores is None:
                raise ValueError("Joint optimization requires stability and binding scores.")
            sequence_indices = {seq: idx for idx, seq in enumerate(sequences)}
            kept_indices = [sequence_indices[c.sequence] for c in kept]
            data["stability_score"] = [float(stability_scores[idx]) for idx in kept_indices]
            data["binding_score"] = [float(binding_scores[idx]) for idx in kept_indices]
            if show_pareto_front:
                pareto_flags = _pareto_front(stability_scores, binding_scores)
                data["pareto_front"] = [bool(pareto_flags[idx]) for idx in kept_indices]

        results[depth] = pd.DataFrame(data)
        current = kept

    _plot_mutation_distributions(results, chain_lengths, plot_dir)
    _plot_pareto_fronts(results, plot_dir)
    _plot_lineage_alluvial(results, plot_dir, top_n=lineage_top_n)
    _plot_mutation_persistence_heatmap(results, plot_dir, top_n=heatmap_top_n)
    _plot_mutation_cooccurrence_networks(
        results,
        plot_dir,
        top_n=cooccurrence_top_n,
        max_edges=cooccurrence_max_edges,
    )
    return results
