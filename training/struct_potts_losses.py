"""Loss utilities for structure↔sequence Potts autoencoder training."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from etab_utils import expand_etab


def expand_etab_dense(etab_geom, e_idx):
    """Expand sparse Potts edges to dense [B, N, N, H]."""
    b, n, k, h = etab_geom.shape
    h2 = int(h**0.5)
    etab_geom_dense = expand_etab(etab_geom.view(b, n, k, h2, h2), e_idx)
    return etab_geom_dense.reshape(b, n, n, h)


def potts_consistency_loss(etab_geom, e_idx, etab_seq_dense, mask, reduction="mean"):
    """Match geometry-derived Potts to sequence-implied Potts."""
    etab_geom_dense = expand_etab_dense(etab_geom, e_idx)
    pair_mask = mask[:, :, None] * mask[:, None, :]
    diff = (etab_geom_dense - etab_seq_dense) * pair_mask[..., None]
    loss = diff.pow(2).sum(dim=-1)
    if reduction == "sum":
        return loss.sum()
    return loss.sum() / (pair_mask.sum() + 1e-6)


def msa_similarity_loss(log_probs, msa_tokens, msa_mask, seq_mask, margin=0.0):
    """Contrastive loss: predicted sequence should score MSA > random decoys."""
    vocab = log_probs.shape[-1]
    msa_tokens = msa_tokens.to(log_probs.device)
    msa_mask = msa_mask.to(log_probs.device)
    seq_mask = seq_mask.to(log_probs.device)

    # Broadcast predicted log-probabilities across the MSA depth.
    log_probs_exp = log_probs[:, None, :, :].expand(
        msa_tokens.shape[0], msa_tokens.shape[1], *log_probs.shape[1:]
    )
    pos_logp = torch.gather(log_probs_exp, -1, msa_tokens[..., None]).squeeze(-1)
    pos_mask = msa_mask * seq_mask[:, None, :]
    pos_loss = -(pos_logp * pos_mask).sum() / (pos_mask.sum() + 1e-6)

    # Use random decoys to keep the signal contrastive without additional inputs.
    decoy_tokens = torch.randint_like(msa_tokens, low=0, high=vocab)
    decoy_logp = torch.gather(log_probs_exp, -1, decoy_tokens[..., None]).squeeze(-1)
    decoy_loss = -(decoy_logp * pos_mask).sum() / (pos_mask.sum() + 1e-6)

    return F.relu(pos_loss - decoy_loss + margin)


def _gumbel_one_hot(log_probs, temperature=1.0, hard=True):
    """Sample straight-through one-hot vectors from log-probabilities."""
    return F.gumbel_softmax(log_probs, tau=temperature, hard=hard, dim=-1)


def _sample_potts_token_ids(log_probs, temperature=1.0):
    """Sample discrete Potts token IDs from log-probabilities."""
    if temperature is None or temperature <= 0:
        return torch.argmax(log_probs, dim=-1)
    one_hot = _gumbel_one_hot(log_probs, temperature=temperature, hard=True)
    return one_hot.argmax(dim=-1)


def _esm_embedding_weight(esm_model):
    for attr in ("embed_tokens", "token_embedder", "embedder"):
        module = getattr(esm_model, attr, None)
        if module is not None and hasattr(module, "weight"):
            return module.weight
    model_attr = getattr(esm_model, "model", None)
    if model_attr is not None:
        module = getattr(model_attr, "embed_tokens", None)
        if module is not None and hasattr(module, "weight"):
            return module.weight
    raise AttributeError("ESM model does not expose an embedding weight.")


def _esm_token_embeddings(esm_model, token_weights, token_id_map):
    """Convert token weights to ESM token embeddings via a token-id mapping."""
    embed_weight = _esm_embedding_weight(esm_model)
    mapped_embed_weight = embed_weight[token_id_map]
    return torch.einsum("blv,ve->ble", token_weights, mapped_embed_weight)


def _esm_token_embeddings_from_ids(esm_model, token_ids, token_id_map):
    """Embed integer token IDs using ESM's embedding table."""
    mapped_ids = token_id_map[token_ids]
    embed_weight = _esm_embedding_weight(esm_model)
    return F.embedding(mapped_ids, embed_weight)


def _esmc_forward_embeddings(esm_model, token_ids, token_mask=None):
    output = esm_model.forward(sequence_tokens=token_ids, sequence_id=token_mask)
    if output.embeddings is not None:
        return output.embeddings
    if output.hidden_states is not None:
        return output.hidden_states[-1]
    raise ValueError("ESM-C forward did not return embeddings or hidden_states.")


def msa_similarity_loss_esmc(
    log_probs,
    msa_tokens,
    msa_mask,
    seq_mask,
    esm_model,
    token_id_map,
    margin=0.0,
    gumbel_temperature=1.0,
):
    """Contrastive ESM-C loss using forward() embeddings."""
    if esm_model is None:
        raise ValueError("msa_similarity_loss_esmc requires a non-null esm_model.")

    device = log_probs.device
    msa_tokens = msa_tokens.to(device)
    msa_mask = msa_mask.to(device)
    seq_mask = seq_mask.to(device)
    token_id_map = token_id_map.to(device)

    pred_one_hot = _gumbel_one_hot(log_probs, temperature=gumbel_temperature, hard=True)
    pred_token_ids = torch.argmax(pred_one_hot, dim=-1)
    pred_token_ids = token_id_map[pred_token_ids]

    pad_id = getattr(getattr(esm_model, "tokenizer", None), "pad_token_id", None)
    pred_mask = seq_mask.bool()
    if pad_id is not None:
        pred_token_ids = pred_token_ids.masked_fill(~pred_mask, int(pad_id))

    pred_embed = _esmc_forward_embeddings(esm_model, pred_token_ids, pred_mask)

    msa_token_ids = token_id_map[msa_tokens]
    msa_mask_full = (msa_mask * seq_mask[:, None, :]).bool()
    if pad_id is not None:
        msa_token_ids = msa_token_ids.masked_fill(~msa_mask_full, int(pad_id))

    b, m, l = msa_token_ids.shape
    msa_embed = _esmc_forward_embeddings(
        esm_model,
        msa_token_ids.reshape(b * m, l),
        msa_mask_full.reshape(b * m, l),
    ).reshape(b, m, l, -1)

    pred_embed = F.normalize(pred_embed, dim=-1)
    msa_embed = F.normalize(msa_embed, dim=-1)
    pred_embed_exp = pred_embed[:, None, :, :]
    sim = (pred_embed_exp * msa_embed).sum(dim=-1)

    pos_mask = msa_mask * seq_mask[:, None, :]
    pos_loss = -(sim * pos_mask).sum() / (pos_mask.sum() + 1e-6)

    vocab = log_probs.shape[-1]
    decoy_tokens = torch.randint_like(msa_tokens, low=0, high=vocab)
    decoy_token_ids = token_id_map[decoy_tokens]
    if pad_id is not None:
        decoy_token_ids = decoy_token_ids.masked_fill(~msa_mask_full, int(pad_id))

    decoy_embed = _esmc_forward_embeddings(
        esm_model,
        decoy_token_ids.reshape(b * m, l),
        msa_mask_full.reshape(b * m, l),
    ).reshape(b, m, l, -1)
    decoy_embed = F.normalize(decoy_embed, dim=-1)
    decoy_sim = (pred_embed_exp * decoy_embed).sum(dim=-1)
    decoy_loss = -(decoy_sim * pos_mask).sum() / (pos_mask.sum() + 1e-6)

    return F.relu(pos_loss - decoy_loss + margin)


def msa_similarity_loss_esm(
    log_probs,
    msa_tokens,
    msa_mask,
    seq_mask,
    esm_model,
    token_id_map,
    margin=0.0,
    gumbel_temperature=1.0,
    pred_embed_mode="gumbel_st",
):
    """Contrastive ESM-embedding loss against MSA sequences.

    """
    if esm_model is None:
        raise ValueError("msa_similarity_loss_esm requires a non-null esm_model.")

    device = log_probs.device
    msa_tokens = msa_tokens.to(device)
    msa_mask = msa_mask.to(device)
    seq_mask = seq_mask.to(device)
    token_id_map = token_id_map.to(device)

    # Map model vocab tokens into ESM's token IDs via the provided lookup.
    if pred_embed_mode == "gumbel_st":
        pred_weights = _gumbel_one_hot(
            log_probs, temperature=gumbel_temperature, hard=True
        )
    elif pred_embed_mode == "potts_weighted":
        pred_weights = log_probs.exp()
    else:
        raise ValueError(
            "pred_embed_mode must be one of ['gumbel_st', 'potts_weighted']."
        )
    pred_embed = _esm_token_embeddings(esm_model, pred_weights, token_id_map)
    msa_embed = _esm_token_embeddings_from_ids(esm_model, msa_tokens, token_id_map)

    # Normalize embeddings for cosine similarity.
    pred_embed = F.normalize(pred_embed, dim=-1)
    msa_embed = F.normalize(msa_embed, dim=-1)

    # Broadcast predicted embeddings across MSA depth.
    pred_embed_exp = pred_embed[:, None, :, :]
    sim = (pred_embed_exp * msa_embed).sum(dim=-1)

    pos_mask = msa_mask * seq_mask[:, None, :]
    pos_loss = -(sim * pos_mask).sum() / (pos_mask.sum() + 1e-6)

    # Contrast with random decoys in token space.
    vocab = log_probs.shape[-1]
    decoy_tokens = torch.randint_like(msa_tokens, low=0, high=vocab)
    if _is_esmc_model(esm_model):
        decoy_embed = _esmc_forward_embeddings(
            esm_model, decoy_tokens, token_id_map
        )
    else:
        decoy_embed = _esm_token_embeddings_from_ids(
            esm_model, decoy_tokens, token_id_map
        )
    decoy_embed = F.normalize(decoy_embed, dim=-1)
    decoy_sim = (pred_embed_exp * decoy_embed).sum(dim=-1)
    decoy_loss = -(decoy_sim * pos_mask).sum() / (pos_mask.sum() + 1e-6)

    return F.relu(pos_loss - decoy_loss + margin)


def structure_consistency_loss(positions, X, mask, atom_index=1):
    """Backbone structure loss using a single atom index (default CA)."""
    if positions is None:
        return torch.tensor(0.0, device=X.device)
    if positions.ndim == 4:
        pred = positions[:, :, atom_index]
    else:
        pred = positions
    target = X[:, :, atom_index]
    diff = (pred - target) * mask[..., None]
    return diff.pow(2).sum() / (mask.sum() + 1e-6)


def structure_fape_loss(frames, backbone_4x4, mask):
    """Backbone FAPE loss using OpenFold's backbone_loss_per_frame."""
    if frames is None or backbone_4x4 is None or backbone_4x4.numel() == 0:
        return torch.tensor(0.0, device=mask.device)
    # OpenFold provides a reference FAPE implementation.
    from openfold.utils.loss import backbone_loss_per_frame

    loss, _ = backbone_loss_per_frame(backbone_4x4, mask, traj=frames[-1])
    if loss.isnan().item():
        return torch.tensor(0.0, device=mask.device)
    return loss
