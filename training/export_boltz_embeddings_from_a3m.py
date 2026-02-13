#!/usr/bin/env python3
"""Export Boltz2 trunk embeddings (s, z) from .a3m MSA files.

This script intentionally avoids the full Boltz prediction pipeline and runs only
trunk/MSA/pairformer computations needed to produce single-site (s) and pairwise
(z) embeddings. It mirrors the write_embeddings payload by saving .npz files with
keys: `s`, `z`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
BOLTZ_SRC = REPO_ROOT / "boltz2" / "src"
if str(BOLTZ_SRC) not in sys.path:
    sys.path.insert(0, str(BOLTZ_SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msa_training.utils import clean_a3m  # noqa: E402
from training.boltz2_adapter import Boltz2TrunkAdapter  # noqa: E402
from training.boltz2_features import build_boltz2_item_feats, collate_boltz2_feats  # noqa: E402
from training.training_struct_potts import load_boltz2_checkpoint  # noqa: E402


def parse_a3m_sequences(a3m_path: Path) -> list[str]:
    """Parse A3M sequences and remove insertion symbols using msa_training.clean_a3m."""
    sequences: list[str] = []
    current: list[str] = []
    with a3m_path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">") or line.startswith("#"):
                if current:
                    seq = clean_a3m("".join(current))
                    if seq:
                        sequences.append(seq)
                    current = []
                continue
            current.append(line)
        if current:
            seq = clean_a3m("".join(current))
            if seq:
                sequences.append(seq)

    if not sequences:
        raise ValueError(f"No sequences parsed from {a3m_path}")

    query_len = len(sequences[0])
    filtered = [s for s in sequences if len(s) == query_len]
    if not filtered:
        raise ValueError(
            f"No sequences with query length ({query_len}) in {a3m_path}"
        )
    return filtered


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


def move_feats_to_device(feats: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in feats.items()}


def export_embeddings(
    a3m_files: Iterable[Path],
    checkpoint: Path,
    out_dir: Path,
    device: torch.device,
    recycling_steps: int,
    output_layout: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_boltz2_checkpoint(str(checkpoint), device)
    model.to(device)
    model.eval()

    trunk = Boltz2TrunkAdapter.from_boltz2_model(model)
    trunk.to(device)
    trunk.eval()

    total = 0
    with torch.no_grad():
        for a3m_path in a3m_files:
            msa_seqs = parse_a3m_sequences(a3m_path)
            query_seq = msa_seqs[0]
            sample_id = a3m_path.stem

            item = {
                "seq": query_seq,
                "chain_order": ["A"],
                "chain_lengths": [len(query_seq)],
                "msa_chain_A": msa_seqs,
            }
            feats_item = build_boltz2_item_feats(item)
            feats_batch = collate_boltz2_feats([feats_item])
            feats_batch = move_feats_to_device(feats_batch, device)

            trunk_out = trunk(feats_batch, recycling_steps=recycling_steps)
            s = trunk_out.s_trunk[0].detach().cpu().numpy().astype(np.float32)
            z = trunk_out.z_trunk[0].detach().cpu().numpy().astype(np.float32)

            if output_layout == "boltz":
                target_dir = out_dir / sample_id
                target_dir.mkdir(parents=True, exist_ok=True)
                out_path = target_dir / f"embeddings_{sample_id}.npz"
            else:
                out_path = out_dir / f"embeddings_{sample_id}.npz"

            np.savez_compressed(out_path, s=s, z=z)
            total += 1
            print(f"[ok] {sample_id}: s{tuple(s.shape)} z{tuple(z.shape)} -> {out_path}")

    print(f"Done. Exported embeddings for {total} A3M file(s).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Boltz2 s/z embeddings directly from .a3m files"
    )
    parser.add_argument(
        "--a3m_input",
        type=Path,
        required=True,
        help="Path to a single .a3m file or a directory containing .a3m files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to Boltz2 checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Output directory for embeddings files.",
    )
    parser.add_argument(
        "--recycling_steps",
        type=int,
        default=3,
        help="Number of trunk recycles to run.",
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
            "'boltz': out_dir/<id>/embeddings_<id>.npz (matches write_embeddings layout); "
            "'flat': out_dir/embeddings_<id>.npz"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    a3m_files = collect_a3m_files(args.a3m_input)
    device = torch.device(args.device)
    export_embeddings(
        a3m_files=a3m_files,
        checkpoint=args.checkpoint,
        out_dir=args.out_dir,
        device=device,
        recycling_steps=args.recycling_steps,
        output_layout=args.output_layout,
    )


if __name__ == "__main__":
    main()
