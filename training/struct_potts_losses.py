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
    module = getattr(esm_model, "embed", None)
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
    embed_weight = _esm_embedding_weight(esm_model)
    embeddings = F.embedding(token_ids, embed_weight)
    return _esmc_forward_from_embeddings(esm_model, embeddings, token_mask)


def _esmc_forward_from_embeddings(esm_model, embeddings, token_mask=None):
    if token_mask is None:
        token_mask = torch.ones(
            embeddings.shape[:2], device=embeddings.device, dtype=torch.bool
        )
    token_mask = token_mask.to(dtype=torch.bool, device=embeddings.device)
    use_flash_attn = getattr(esm_model, "_use_flash_attn", False)
    if use_flash_attn:
        from esm.models.esmc import pad_input, unpad_input

        x, indices, *_ = unpad_input(embeddings, token_mask)
        x, _, _ = esm_model.transformer(x, sequence_id=token_mask)
        x = pad_input(x, indices, embeddings.shape[0], embeddings.shape[1])
    else:
        x, _, _ = esm_model.transformer(embeddings, sequence_id=token_mask)
    return x


def _is_esmc_model(esm_model):
    return hasattr(esm_model, "tokenizer") and hasattr(esm_model, "sequence_head")


class ESMCDecoySequencePool:
    """Reservoir-like pool of real MSA sequences with cached ESM-C input embeddings."""

    def __init__(self, max_size=8192):
        self.max_size = int(max_size)
        self._entries_by_len = {}

    def _bucket(self, length):
        bucket = self._entries_by_len.get(length)
        if bucket is None:
            bucket = {}
            self._entries_by_len[length] = bucket
        return bucket

    @staticmethod
    def _to_key(token_ids_1d):
        return token_ids_1d.detach().to(torch.int16).cpu().numpy().tobytes()

    @torch.no_grad()
    def add(self, mapped_token_ids, embed_in=None):
        """Add tokenized sequences [N, L] (and optional embedding inputs [N, L, E])."""
        if mapped_token_ids.ndim != 2:
            raise ValueError("mapped_token_ids must have shape [N, L].")
        n, length = mapped_token_ids.shape
        bucket = self._bucket(length)
        mapped_token_ids_cpu = mapped_token_ids.detach().to(torch.long).cpu()
        embed_cpu = None
        if embed_in is not None:
            embed_cpu = embed_in.detach().cpu()
        for i in range(n):
            key = self._to_key(mapped_token_ids_cpu[i])
            if key in bucket:
                continue
            bucket[key] = {
                "token_ids": mapped_token_ids_cpu[i].clone(),
                "embed": embed_cpu[i].clone() if embed_cpu is not None else None,
            }
            if len(bucket) > self.max_size:
                # Random eviction keeps memory bounded without expensive bookkeeping.
                victim_key = next(iter(bucket))
                del bucket[victim_key]

    def sample(self, length, n_samples, exclude_token_ids=None, device=None, dtype=None):
        """Sample up to n_samples sequences for a specific length, excluding provided IDs."""
        bucket = self._entries_by_len.get(length, {})
        if not bucket or n_samples <= 0:
            return None, None

        exclude_keys = set()
        if exclude_token_ids is not None:
            exclude_ids = exclude_token_ids.detach().to(torch.long).cpu().reshape(-1, length)
            exclude_keys = {self._to_key(row) for row in exclude_ids}

        available = [k for k in bucket.keys() if k not in exclude_keys]
        if not available:
            return None, None
        n_take = min(int(n_samples), len(available))
        pick_idx = torch.randperm(len(available))[:n_take].tolist()
        picks = [bucket[available[i]] for i in pick_idx]

        token_ids = torch.stack([p["token_ids"] for p in picks], dim=0)
        embeds = None
        if picks[0]["embed"] is not None:
            embeds = torch.stack([p["embed"] for p in picks], dim=0)
            if dtype is not None:
                embeds = embeds.to(dtype=dtype)
        if device is not None:
            token_ids = token_ids.to(device)
            if embeds is not None:
                embeds = embeds.to(device)
        return token_ids, embeds


def msa_similarity_loss_esmc(
    log_probs,
    msa_tokens,
    msa_mask,
    seq_mask,
    esm_model,
    token_id_map,
    margin=0.0,
    gumbel_temperature=1.0,
    decoy_real_fraction=0.0,
    decoy_pool=None,
):
    """Contrastive ESM-C loss using forward() embeddings."""
    if esm_model is None:
        raise ValueError("msa_similarity_loss_esmc requires a non-null esm_model.")

    device = log_probs.device
    msa_tokens = msa_tokens.to(device)
    msa_mask = msa_mask.to(device)
    seq_mask = seq_mask.to(device)
    token_id_map = token_id_map.to(device)

    pred_one_hot = _gumbel_one_hot(
        log_probs, temperature=gumbel_temperature, hard=True
    )
    embed_weight = _esm_embedding_weight(esm_model)
    mapped_embed_weight = embed_weight[token_id_map]
    pred_one_hot = pred_one_hot.to(dtype=mapped_embed_weight.dtype)
    pred_mask = seq_mask.bool()
    pred_embed_in = torch.einsum("blv,ve->ble", pred_one_hot, mapped_embed_weight)
    pred_embed_in = pred_embed_in * pred_mask[..., None]

    pad_id = getattr(getattr(esm_model, "tokenizer", None), "pad_token_id", None)
    msa_token_ids = token_id_map[msa_tokens]
    msa_mask_full = (msa_mask * seq_mask[:, None, :]).bool()
    if pad_id is not None:
        msa_token_ids = msa_token_ids.masked_fill(~msa_mask_full, int(pad_id))

    b, m, l = msa_token_ids.shape
    msa_token_ids = msa_token_ids.reshape(b * m, l)
    msa_mask_flat = msa_mask_full.reshape(b * m, l)
    msa_embed_in = F.embedding(msa_token_ids, embed_weight)

    if decoy_pool is not None:
        decoy_pool.add(msa_token_ids, msa_embed_in)

    vocab = log_probs.shape[-1]
    decoy_tokens = torch.randint_like(msa_tokens, low=0, high=vocab)
    decoy_token_ids = token_id_map[decoy_tokens]
    if pad_id is not None:
        decoy_token_ids = decoy_token_ids.masked_fill(~msa_mask_full, int(pad_id))
    decoy_token_ids = decoy_token_ids.reshape(b * m, l)
    decoy_embed_in = F.embedding(decoy_token_ids, embed_weight)

    real_fraction = float(max(0.0, min(1.0, decoy_real_fraction)))
    n_real = int(round(real_fraction * decoy_token_ids.shape[0]))
    if decoy_pool is not None and n_real > 0:
        sampled_ids, sampled_embed = decoy_pool.sample(
            length=l,
            n_samples=n_real,
            exclude_token_ids=msa_token_ids,
            device=device,
            dtype=embed_weight.dtype,
        )
        if sampled_ids is not None:
            take = sampled_ids.shape[0]
            perm = torch.randperm(decoy_token_ids.shape[0], device=device)[:take]
            decoy_token_ids[perm] = sampled_ids
            if sampled_embed is None:
                sampled_embed = F.embedding(sampled_ids, embed_weight)
            decoy_embed_in[perm] = sampled_embed

    combined_embed_in = torch.cat([pred_embed_in, msa_embed_in, decoy_embed_in], dim=0)
    combined_mask = torch.cat([pred_mask, msa_mask_flat, msa_mask_flat], dim=0)
    combined_embed = _esmc_forward_from_embeddings(
        esm_model, combined_embed_in, combined_mask
    )

    pred_embed = combined_embed[:b]
    msa_embed = combined_embed[b : b + b * m].reshape(b, m, l, -1)
    decoy_embed = combined_embed[b + b * m :].reshape(b, m, l, -1)

    pred_embed = F.normalize(pred_embed, dim=-1)
    msa_embed = F.normalize(msa_embed, dim=-1)
    pred_embed_exp = pred_embed[:, None, :, :]
    sim = (pred_embed_exp * msa_embed).sum(dim=-1)

    pos_mask = msa_mask * seq_mask[:, None, :]
    pos_loss = -(sim * pos_mask).sum() / (pos_mask.sum() + 1e-6)

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
        raise ValueError(
            "Use msa_similarity_loss_esmc for ESM-C models instead of msa_similarity_loss_esm."
        )
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
