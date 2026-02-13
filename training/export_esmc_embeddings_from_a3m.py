#!/usr/bin/env python3
"""Export ESM-C embeddings from .a3m MSA files.

This script parses A3M files, applies the same sequence filtering used in
`export_boltz_embeddings_from_a3m.py`, and then embeds kept sequences with an
ESM-C model. Embeddings are written as compressed `.npz` files.
"""

from __future__ import annotations

import argparse
import sys
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
            emb = output.embeddings[:, 1:-1, :].detach().cpu().numpy().astype(np.float32)
            chunks.append(emb)

    return np.concatenate(chunks, axis=0)


def export_embeddings(
    a3m_files: Iterable[Path],
    model_name: str,
    out_dir: Path,
    device: torch.device,
    output_layout: str,
    id_thresh: float,
    del_thresh: float,
    insrt_thresh: float,
    max_msa_seqs: int,
    batch_size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ESMC.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    total = 0
    for a3m_path in tqdm(a3m_files):
        msa_seqs = parse_a3m_sequences(
            a3m_path,
            id_thresh=id_thresh,
            del_thresh=del_thresh,
            insrt_thresh=insrt_thresh,
        )
        if max_msa_seqs > 0:
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
    a3m_files = collect_a3m_files(args.a3m_input)
    device = torch.device(args.device)
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
    )


if __name__ == "__main__":
    main()
