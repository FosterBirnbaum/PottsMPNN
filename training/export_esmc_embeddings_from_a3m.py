#!/usr/bin/env python3
"""Export ESM-C embeddings from .a3m MSA files.

This script parses A3M files, applies the same sequence filtering used in
`export_boltz_embeddings_from_a3m.py`, and then embeds kept sequences with an
ESM-C model. Embeddings are written as compressed `.npz` files.
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
import sys
from bisect import bisect_left
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
ESM_SRC = REPO_ROOT / "esm"
if str(ESM_SRC) not in sys.path:
    sys.path.insert(0, str(ESM_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msa_training.utils import clean_a3m  # noqa: E402
from esm.models.esmc import ESMC  # noqa: E402
from esm.utils.encoding import tokenize_sequence  # noqa: E402


AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
# Approximate natural amino-acid frequencies (sum ~= 1.0), used to bias random
# draws toward biologically plausible compositions instead of uniform noise.
AMINO_ACID_FREQS = (
    0.083,  # A
    0.014,  # C
    0.054,  # D
    0.067,  # E
    0.039,  # F
    0.072,  # G
    0.022,  # H
    0.057,  # I
    0.058,  # K
    0.097,  # L
    0.024,  # M
    0.040,  # N
    0.047,  # P
    0.039,  # Q
    0.052,  # R
    0.068,  # S
    0.058,  # T
    0.073,  # V
    0.013,  # W
    0.033,  # Y
)


def parse_a3m_sequences(
    a3m_path: Path,
    id_thresh: float,
    del_thresh: float,
    insrt_thresh: float,
) -> list[str]:
    """Parse and filter A3M sequences.

    Keep sequences only if:
      - sequence identity to query > id_thresh
      - deletion percent <= del_thresh
      - insertion percent <= insrt_thresh
    """
    raw_sequences: list[str] = []
    current: list[str] = []
    with a3m_path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">") or line.startswith("#"):
                if current:
                    raw_sequences.append("".join(current))
                    current = []
                continue
            current.append(line)
        if current:
            raw_sequences.append("".join(current))

    if not raw_sequences:
        raise ValueError(f"No sequences parsed from {a3m_path}")

    ref_seq = clean_a3m(raw_sequences[0])
    query_len = len(ref_seq)
    kept_sequences: list[str] = [ref_seq]

    for seq_raw in raw_sequences[1:]:
        if any(ch in seq_raw for ch in "BJOUZ"):
            continue
        insert_pct = len([ch for ch in seq_raw if ch.islower()]) / len(seq_raw) if seq_raw else 0.0
        seq = clean_a3m(seq_raw)
        if len(seq) != query_len:
            continue

        aligned_len = min(len(ref_seq), len(seq))
        identical = sum(ref_seq[j] == seq[j] for j in range(aligned_len))
        identity = identical / aligned_len if aligned_len > 0 else 0.0
        deletion_pct = seq.count("-") / len(seq) if seq else 0.0

        if identity <= id_thresh or deletion_pct > del_thresh or insert_pct > insrt_thresh:
            continue
        kept_sequences.append(seq)

    if not kept_sequences:
        raise ValueError(f"No sequences remained after filtering in {a3m_path}")
    return kept_sequences


def collect_a3m_files(a3m_input: Path) -> list[Path]:
    if a3m_input.is_file():
        if a3m_input.suffix != ".a3m":
            raise ValueError(f"Input file must end in .a3m, got: {a3m_input}")
        return [a3m_input]
    if not a3m_input.exists():
        raise FileNotFoundError(a3m_input)
    files = sorted(a3m_input.glob("*.a3m"))
    if not files:
        raise ValueError(f"No .a3m files found in {a3m_input}")
    return files


def parse_query_sequence(a3m_path: Path) -> str:
    """Parse only the native/query sequence from an A3M file."""
    current: list[str] = []
    with a3m_path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">") or line.startswith("#"):
                if current:
                    break
                continue
            current.append(line)

    if not current:
        raise ValueError(f"No query sequence parsed from {a3m_path}")

    return clean_a3m("".join(current))


def parse_query_length(a3m_path: Path) -> int:
    """Parse only the native/query sequence length from an A3M file."""
    return len(parse_query_sequence(a3m_path))


def collect_unique_query_lengths(a3m_files: Iterable[Path]) -> list[int]:
    """Collect sorted unique native/query lengths across provided A3M files."""
    lengths = {parse_query_length(a3m_path) for a3m_path in a3m_files}
    if not lengths:
        raise ValueError("No query lengths found in provided A3M files.")
    return sorted(lengths)


def collect_length_clusters(query_sequences: dict[str, str]) -> dict[int, set[str]]:
    """Group protein names by query length."""
    clusters: dict[int, set[str]] = {}
    for protein_name, sequence in query_sequences.items():
        clusters.setdefault(len(sequence), set()).add(protein_name)

    if not clusters:
        raise ValueError("No length clusters found in provided A3M files.")
    return clusters


def collect_query_sequences(a3m_files: Iterable[Path]) -> dict[str, str]:
    """Parse query sequence from each A3M file keyed by protein name (file stem)."""
    query_sequences: dict[str, str] = {}
    for a3m_path in a3m_files:
        protein_name = a3m_path.stem
        query_sequences[protein_name] = parse_query_sequence(a3m_path)

    if not query_sequences:
        raise ValueError("No query sequences found in provided A3M files.")
    return query_sequences


def is_identity_below_threshold(seq_a: str, seq_b: str, threshold: float = 0.5) -> bool:
    """Return True if identity(seq_a, seq_b) is strictly below threshold.

    Identity is computed over min(len(seq_a), len(seq_b)).
    """
    aligned_len = min(len(seq_a), len(seq_b))
    if aligned_len == 0:
        return True

    # Need identity < threshold. If matches >= cutoff, condition fails.
    cutoff_matches = math.ceil(threshold * aligned_len)
    matches = 0
    for idx in range(aligned_len):
        if seq_a[idx] == seq_b[idx]:
            matches += 1
            if matches >= cutoff_matches:
                return False
        # even with all remaining matches, cannot hit cutoff => definitely below threshold
        remaining = aligned_len - idx - 1
        if matches + remaining < cutoff_matches:
            return True

    return True


def _iter_candidate_lengths_by_distance(target_length: int, sorted_lengths: list[int]) -> Iterable[list[int]]:
    """Yield candidate lengths grouped by increasing absolute distance."""
    insert_idx = bisect_left(sorted_lengths, target_length)
    left = insert_idx - 1
    right = insert_idx

    while left >= 0 or right < len(sorted_lengths):
        left_dist = abs(target_length - sorted_lengths[left]) if left >= 0 else None
        right_dist = abs(sorted_lengths[right] - target_length) if right < len(sorted_lengths) else None

        if left_dist is None:
            current_dist = right_dist
        elif right_dist is None:
            current_dist = left_dist
        else:
            current_dist = min(left_dist, right_dist)

        candidates: list[int] = []
        while left >= 0 and abs(target_length - sorted_lengths[left]) == current_dist:
            candidates.append(sorted_lengths[left])
            left -= 1
        while right < len(sorted_lengths) and abs(sorted_lengths[right] - target_length) == current_dist:
            candidates.append(sorted_lengths[right])
            right += 1

        if candidates:
            yield candidates


def build_valid_protein_map(
    query_sequences: dict[str, str],
    length_clusters: dict[int, set[str]],
    similarity_threshold: float = 0.5,
) -> dict[str, list[str]]:
    """Build protein->valid_proteins map according to length/similarity rules."""
    sequence_by_name = query_sequences
    names_by_length = {length: sorted(names) for length, names in length_clusters.items()}

    valid_map: dict[str, list[str]] = {}

    # Case 1: proteins with peers at same length -> evaluate only that cluster.
    for length, names in tqdm(names_by_length.items(), desc="same-length valid maps"):
        if len(names) < 2:
            continue

        seqs = [sequence_by_name[name] for name in names]
        neighbors: dict[str, list[str]] = {name: [] for name in names}
        for i in range(len(names)):
            seq_i = seqs[i]
            for j in range(i + 1, len(names)):
                if is_identity_below_threshold(seq_i, seqs[j], threshold=similarity_threshold):
                    neighbors[names[i]].append(names[j])
                    neighbors[names[j]].append(names[i])

        for name in names:
            valid_map[name] = neighbors[name]

    # Case 2: proteins with no same-length peers -> search nearest lengths with valid identities.
    sorted_lengths = sorted(names_by_length)
    singleton_lengths = [length for length, names in names_by_length.items() if len(names) == 1]
    for length in tqdm(singleton_lengths, desc="nearest-length valid maps"):
        name = names_by_length[length][0]
        seq = sequence_by_name[name]

        best_valids: list[str] = []
        for candidate_lengths in _iter_candidate_lengths_by_distance(length, sorted_lengths):
            valid_here: list[str] = []
            for candidate_length in candidate_lengths:
                if candidate_length == length:
                    continue
                for candidate_name in names_by_length[candidate_length]:
                    if is_identity_below_threshold(
                        seq,
                        sequence_by_name[candidate_name],
                        threshold=similarity_threshold,
                    ):
                        valid_here.append(candidate_name)

            if valid_here:
                best_valids = valid_here
                break

        valid_map[name] = best_valids

    # Ensure every protein key exists.
    for name in sequence_by_name:
        valid_map.setdefault(name, [])

    return valid_map


def save_length_cluster_artifacts(
    a3m_files: list[Path],
    out_dir: Path,
    cluster_output_name: str,
    valid_pairs_output_name: str,
    similarity_threshold: float,
) -> tuple[Path, Path]:
    """Save length clusters and valid protein map as pickles."""
    out_dir.mkdir(parents=True, exist_ok=True)

    query_sequences = collect_query_sequences(a3m_files)
    length_clusters = collect_length_clusters(query_sequences)
    valid_proteins = build_valid_protein_map(
        query_sequences=query_sequences,
        length_clusters=length_clusters,
        similarity_threshold=similarity_threshold,
    )

    cluster_out_path = out_dir / cluster_output_name
    with cluster_out_path.open("wb") as handle:
        pickle.dump(length_clusters, handle)

    valid_out_path = out_dir / valid_pairs_output_name
    with valid_out_path.open("wb") as handle:
        pickle.dump(valid_proteins, handle)

    print(
        f"Done. Saved {len(length_clusters)} length cluster(s) and valid partners for "
        f"{len(valid_proteins)} protein(s) from {len(a3m_files)} A3M file(s)."
    )
    print(f"  length clusters -> {cluster_out_path}")
    print(f"  valid partners -> {valid_out_path}")

    return cluster_out_path, valid_out_path


def generate_random_sequences(length: int, count: int, rng: random.Random) -> list[str]:
    """Generate random but biologically plausible protein sequences."""
    if length <= 0:
        raise ValueError("length must be > 0")
    if count <= 0:
        raise ValueError("count must be > 0")

    sequences: list[str] = []
    for _ in range(count):
        seq_chars = rng.choices(AMINO_ACIDS, weights=AMINO_ACID_FREQS, k=length)
        sequences.append("".join(seq_chars))
    return sequences


def embed_sequences(
    model: ESMC,
    sequences: list[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Return residue embeddings with shape [N, L, D]."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    tokenizer = model.tokenizer
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            tokens = torch.stack(
                [tokenize_sequence(seq, tokenizer, add_special_tokens=True) for seq in batch],
                dim=0,
            ).to(device)
            output = model(sequence_tokens=tokens)
            if output.embeddings is None:
                raise RuntimeError("ESM-C forward returned no embeddings")
            # Remove BOS/EOS so stored embeddings match residue length.
            emb = output.embeddings[:, 1:-1, :].detach().cpu().float().numpy()
            chunks.append(emb)

    return np.concatenate(chunks, axis=0)


def export_embeddings(
    a3m_files: list[Path],
    model_name: str,
    out_dir: Path,
    device: torch.device,
    output_layout: str,
    id_thresh: float,
    del_thresh: float,
    insrt_thresh: float,
    max_msa_seqs: int,
    batch_size: int,
    use_random_sequences: bool,
    num_random_sequences: int,
    random_seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ESMC.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    total = 0
    rng = random.Random(random_seed)

    if use_random_sequences:
        unique_lengths = collect_unique_query_lengths(a3m_files)
        for query_len in tqdm(unique_lengths):
            sample_id = f"len_{query_len}"
            msa_seqs = generate_random_sequences(
                length=query_len,
                count=num_random_sequences,
                rng=rng,
            )
            msa_embeddings = embed_sequences(model, msa_seqs, device=device, batch_size=batch_size)
            query_embedding = msa_embeddings[0]

            if output_layout == "boltz":
                target_dir = out_dir / sample_id
                target_dir.mkdir(parents=True, exist_ok=True)
                out_path = target_dir / f"embeddings_{sample_id}.npz"
            else:
                out_path = out_dir / f"embeddings_{sample_id}.npz"

            np.savez_compressed(
                out_path,
                query_embedding=query_embedding,
                msa_embeddings=msa_embeddings,
                sequences=np.asarray(msa_seqs, dtype=object),
            )
            total += 1
            print(
                f"[ok] {sample_id}: random_seqs={len(msa_seqs)} "
                f"query{tuple(query_embedding.shape)} msa{tuple(msa_embeddings.shape)} -> {out_path}"
            )

        print(
            f"Done. Exported embeddings for {total} unique native length(s) "
            f"from {len(a3m_files)} A3M file(s)."
        )
        return

    for a3m_path in tqdm(a3m_files):
        parsed_msa_seqs = parse_a3m_sequences(
            a3m_path,
            id_thresh=id_thresh,
            del_thresh=del_thresh,
            insrt_thresh=insrt_thresh,
        )
        msa_seqs = parsed_msa_seqs
        if max_msa_seqs > 0:
            native_seq = msa_seqs[0]
            non_native_seqs = msa_seqs[1:]
            shuffle_rng = np.random.default_rng()
            shuffle_rng.shuffle(non_native_seqs)
            msa_seqs = [native_seq, *non_native_seqs]
            msa_seqs = msa_seqs[:max_msa_seqs]

        sample_id = a3m_path.stem
        msa_embeddings = embed_sequences(model, msa_seqs, device=device, batch_size=batch_size)
        query_embedding = msa_embeddings[0]

        if output_layout == "boltz":
            target_dir = out_dir / sample_id
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / f"embeddings_{sample_id}.npz"
        else:
            out_path = out_dir / f"embeddings_{sample_id}.npz"

        np.savez_compressed(
            out_path,
            query_embedding=query_embedding,
            msa_embeddings=msa_embeddings,
            sequences=np.asarray(msa_seqs, dtype=object),
        )
        total += 1
        print(
            f"[ok] {sample_id}: kept_msa={len(msa_seqs)} "
            f"(id>{id_thresh}, del<={del_thresh}, ins<={insrt_thresh}) "
            f"query{tuple(query_embedding.shape)} msa{tuple(msa_embeddings.shape)} -> {out_path}"
        )

    print(f"Done. Exported embeddings for {total} A3M file(s).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ESM-C embeddings from .a3m files"
    )
    parser.add_argument(
        "--a3m_input",
        type=Path,
        required=True,
        help="Path to a single .a3m file or a directory containing .a3m files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for embedding files.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="esmc_300m",
        help="ESM-C pretrained model name (e.g., esmc_300m, esmc_600m).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of sequences per ESM-C forward pass.",
    )
    parser.add_argument(
        "--max_msa_seqs",
        type=int,
        default=0,
        help="Maximum kept MSA depth to embed (0 means no limit).",
    )
    parser.add_argument(
        "--use_random_sequences",
        action="store_true",
        help=(
            "If set, ignore MSA sequences and instead: collect unique native/query "
            "lengths across all .a3m files, then embed random sequences for each length."
        ),
    )
    parser.add_argument(
        "--save_length_clusters_only",
        action="store_true",
        help=(
            "If set, skip ESM-C embedding and save dictionaries for length clusters and "
            "valid protein partners by length/similarity rules."
        ),
    )
    parser.add_argument(
        "--length_clusters_filename",
        type=str,
        default="length_clusters.pkl",
        help="Output filename for length-cluster dictionary when --save_length_clusters_only is set.",
    )
    parser.add_argument(
        "--valid_proteins_filename",
        type=str,
        default="valid_proteins.pkl",
        help="Output filename for valid-proteins dictionary when --save_length_clusters_only is set.",
    )
    parser.add_argument(
        "--valid_similarity_threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for valid protein pairing (default: identity < 0.5).",
    )
    parser.add_argument(
        "--num_random_sequences",
        type=int,
        default=128,
        help="Number of random sequences to embed per input .a3m when random mode is enabled.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed used for random sequence generation mode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="Torch device string (e.g., cpu, cuda:0).",
    )
    parser.add_argument(
        "--output_layout",
        choices=["boltz", "flat"],
        default="boltz",
        help=(
            "'boltz': out_dir/<id>/embeddings_<id>.npz; "
            "'flat': out_dir/embeddings_<id>.npz"
        ),
    )
    parser.add_argument(
        "--id_thresh",
        type=float,
        default=0.5,
        help="Sequence identity cutoff for MSA sequences (keep if identity > cutoff).",
    )
    parser.add_argument(
        "--del_thresh",
        type=float,
        default=0.2,
        help="Gap/deletion percentage cutoff (keep if deletion_pct <= cutoff).",
    )
    parser.add_argument(
        "--insrt_thresh",
        type=float,
        default=0.2,
        help="Insertion percentage cutoff (keep if insertion_pct <= cutoff).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.save_length_clusters_only and args.use_random_sequences:
        raise ValueError("--save_length_clusters_only cannot be combined with --use_random_sequences.")
    if args.use_random_sequences and args.num_random_sequences <= 0:
        raise ValueError("--num_random_sequences must be > 0 when --use_random_sequences is set.")
    if not (0.0 < args.valid_similarity_threshold <= 1.0):
        raise ValueError("--valid_similarity_threshold must be in (0, 1].")

    a3m_files = collect_a3m_files(args.a3m_input)
    if args.save_length_clusters_only:
        print(
            f"Length-cluster-only mode enabled: parsing {len(a3m_files)} A3M file(s) and saving to {args.out_dir}."
        )
        save_length_cluster_artifacts(
            a3m_files=a3m_files,
            out_dir=args.out_dir,
            cluster_output_name=args.length_clusters_filename,
            valid_pairs_output_name=args.valid_proteins_filename,
            similarity_threshold=args.valid_similarity_threshold,
        )
        return

    device = torch.device(args.device)
    if args.use_random_sequences:
        print(
            f"Random sequence mode enabled: generating {args.num_random_sequences} random sequences "
            f"per unique native length across {len(a3m_files)} A3M file(s) with seed={args.random_seed}."
        )
    else:
        print(
            f"Exporting ESM-C embeddings from {len(a3m_files)} A3M file(s) "
            f"using model={args.model_name} and saving to {args.out_dir}."
        )
    export_embeddings(
        a3m_files=a3m_files,
        model_name=args.model_name,
        out_dir=args.out_dir,
        device=device,
        output_layout=args.output_layout,
        id_thresh=args.id_thresh,
        del_thresh=args.del_thresh,
        insrt_thresh=args.insrt_thresh,
        max_msa_seqs=args.max_msa_seqs,
        batch_size=args.batch_size,
        use_random_sequences=args.use_random_sequences,
        num_random_sequences=args.num_random_sequences,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
