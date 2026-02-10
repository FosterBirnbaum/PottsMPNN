"""Build lightweight Boltz2 feature dicts from ProteinMPNN-style batches."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch.nn.functional import one_hot

from boltz.data import const
from boltz.data.feature.featurizerv2 import convert_atom_name, load_dummy_templates_features
from boltz.data.pad import pad_to_max


_CA_ATOM_NAME = "CA"
_CA_ATOM_ENCODED = torch.tensor(convert_atom_name(_CA_ATOM_NAME), dtype=torch.long)
_BACKBONE_DIM = (
    1
    + len(const.protein_backbone_atom_index)
    + len(const.nucleic_backbone_atom_index)
)


def _token_ids_from_sequence(seq: str) -> torch.Tensor:
    tokens = []
    for aa in seq:
        if aa == "-":
            token = "-"
        else:
            token = const.prot_letter_to_token.get(aa.upper(), "UNK")
        tokens.append(const.token_ids[token])
    return torch.tensor(tokens, dtype=torch.long)


def _normalize_msa_sequences(seqs: Iterable[str], length: int, fallback: str) -> list[str]:
    normalized = [s for s in seqs if len(s) == length]
    if not normalized:
        normalized = [fallback]
    return normalized


def _combine_chain_msas(chain_msas: list[list[str]]) -> list[str]:
    if not chain_msas:
        return []
    depth = max(len(msa) for msa in chain_msas)
    combined = []
    for idx in range(depth):
        parts = []
        for msa in chain_msas:
            pick = msa[idx] if idx < len(msa) else msa[0]
            parts.append(pick)
        combined.append("".join(parts))
    return combined


def build_boltz2_item_feats(item: dict) -> dict[str, torch.Tensor]:
    """Build Boltz2-compatible features for a single ProteinMPNN sample."""
    seq = item["seq"]
    length = len(seq)
    chain_order = item.get("chain_order", [])
    chain_lengths = item.get("chain_lengths", [])
    if not chain_order or not chain_lengths:
        chain_order = ["A"]
        chain_lengths = [length]

    res_token_ids = _token_ids_from_sequence(seq)
    res_type = one_hot(res_token_ids, num_classes=const.num_tokens)

    chain_msas = []
    cursor = 0
    for chain_id, chain_len in zip(chain_order, chain_lengths):
        chain_seq = seq[cursor : cursor + chain_len]
        cursor += chain_len
        raw_msa = item.get(f"msa_chain_{chain_id}")
        if raw_msa is None:
            raw_msa = item.get(f"seq_chain_{chain_id}", chain_seq)
        if isinstance(raw_msa, str):
            raw_msa = [raw_msa]
        chain_msas.append(_normalize_msa_sequences(raw_msa, chain_len, chain_seq))

    combined_msa = _combine_chain_msas(chain_msas)
    if not combined_msa:
        combined_msa = [seq]

    msa_tokens = torch.stack([_token_ids_from_sequence(s) for s in combined_msa], dim=0)
    msa_mask = torch.ones_like(msa_tokens, dtype=torch.float)
    msa_one_hot = one_hot(msa_tokens, num_classes=const.num_tokens).float()
    profile = msa_one_hot.mean(dim=0)
    deletion_mean = torch.zeros(length, dtype=torch.float)

    has_deletion = torch.zeros_like(msa_tokens, dtype=torch.float)
    deletion_value = torch.zeros_like(msa_tokens, dtype=torch.float)
    msa_paired = torch.zeros_like(msa_tokens, dtype=torch.float)

    residue_index = []
    asym_id = []
    entity_id = []
    sym_id = []
    cursor = 0
    for chain_idx, chain_len in enumerate(chain_lengths):
        residue_index.extend(range(chain_len))
        asym_id.extend([chain_idx] * chain_len)
        entity_id.extend([chain_idx] * chain_len)
        sym_id.extend([0] * chain_len)
        cursor += chain_len

    residue_index = torch.tensor(residue_index, dtype=torch.long)
    asym_id = torch.tensor(asym_id, dtype=torch.long)
    entity_id = torch.tensor(entity_id, dtype=torch.long)
    sym_id = torch.tensor(sym_id, dtype=torch.long)

    mol_type = torch.full((length,), const.chain_type_ids["PROTEIN"], dtype=torch.long)
    modified = torch.zeros(length, dtype=torch.long)
    method_feature = torch.zeros(length, dtype=torch.long)
    cyclic_period = torch.zeros(length, dtype=torch.float)

    token_index = torch.arange(length, dtype=torch.long)
    token_pad_mask = torch.ones(length, dtype=torch.float)

    contact_conditioning = torch.full(
        (length, length),
        const.contact_conditioning_info["UNSPECIFIED"],
        dtype=torch.long,
    )
    contact_conditioning = one_hot(
        contact_conditioning, num_classes=len(const.contact_conditioning_info)
    )
    contact_threshold = torch.zeros((length, length), dtype=torch.float)

    token_bonds = torch.zeros((length, length, 1), dtype=torch.float)
    type_bonds = torch.zeros((length, length), dtype=torch.long)

    atom14_xyz = item.get("atom14_xyz")
    atom14_mask = item.get("atom14_mask")
    if atom14_xyz is None:
        atom14_xyz = np.zeros((length, 14, 3), dtype=np.float32)
    if atom14_mask is None:
        atom14_mask = np.ones((length, 14), dtype=np.float32)

    atom14_xyz = torch.tensor(atom14_xyz, dtype=torch.float)
    atom14_mask = torch.tensor(atom14_mask, dtype=torch.float)
    ca_coords = atom14_xyz[:, 1, :]
    atom_pad_mask = atom14_mask[:, 1].clone()

    ref_pos = ca_coords
    ref_charge = torch.zeros(length, dtype=torch.float)
    ref_element = one_hot(
        torch.full((length,), 6, dtype=torch.long), num_classes=const.num_elements
    )
    ref_atom_name_chars = one_hot(
        _CA_ATOM_ENCODED.expand(length, -1), num_classes=64
    )
    ref_space_uid = torch.arange(length, dtype=torch.long)

    backbone_feat_index = torch.full(
        (length,),
        const.protein_backbone_atom_index[_CA_ATOM_NAME] + 1,
        dtype=torch.long,
    )
    atom_backbone_feat = one_hot(backbone_feat_index, num_classes=_BACKBONE_DIM)
    print('length: ', length)
    print('seq: ', seq)
    atom_to_token = one_hot(
        torch.arange(length, dtype=torch.long), num_classes=length
    ).float()

    template_features = load_dummy_templates_features(tdim=1, num_tokens=length)

    return {
        "res_type": res_type,
        "profile": profile,
        "deletion_mean": deletion_mean,
        "msa": msa_tokens,
        "msa_mask": msa_mask,
        "has_deletion": has_deletion,
        "deletion_value": deletion_value,
        "msa_paired": msa_paired,
        "token_index": token_index,
        "residue_index": residue_index,
        "asym_id": asym_id,
        "entity_id": entity_id,
        "sym_id": sym_id,
        "mol_type": mol_type,
        "modified": modified,
        "method_feature": method_feature,
        "cyclic_period": cyclic_period,
        "token_pad_mask": token_pad_mask,
        "contact_conditioning": contact_conditioning,
        "contact_threshold": contact_threshold,
        "token_bonds": token_bonds,
        "type_bonds": type_bonds,
        "ref_pos": ref_pos,
        "ref_charge": ref_charge,
        "ref_element": ref_element,
        "ref_atom_name_chars": ref_atom_name_chars,
        "ref_space_uid": ref_space_uid,
        "atom_backbone_feat": atom_backbone_feat,
        "atom_pad_mask": atom_pad_mask,
        "atom_to_token": atom_to_token,
        **template_features,
    }


_PAD_VALUES = {
    "msa": const.token_ids["-"],
}


def collate_boltz2_feats(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad and stack Boltz2 feature dicts into batch tensors."""
    keys = items[0].keys()
    collated = {}
    for key in keys:
        values = [item[key] for item in items]
        if not torch.is_tensor(values[0]):
            collated[key] = values
            continue
        pad_value = _PAD_VALUES.get(key, 0)
        padded, _ = pad_to_max(values, value=pad_value)
        collated[key] = padded
    return collated
