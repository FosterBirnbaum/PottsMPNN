"""Adapters to reuse Boltz2 trunk outputs inside PottsMPNN-style training loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from boltz.model.modules.trunkv2 import ContactConditioning, InputEmbedder
from boltz.model.modules.encodersv2 import RelativePositionEncoder
from boltz.model.layers.pairformer import PairformerModule
from boltz.model.modules.trunkv2 import MSAModule

from etab_utils import expand_etab


@dataclass
class TrunkOutputs:
    """Container for Boltz2 trunk outputs used in PottsMPNN integration."""

    s_trunk: Tensor
    z_trunk: Tensor
    mask: Tensor
    pair_mask: Tensor
    s_inputs: Tensor
    z_init: Tensor


class Boltz2TrunkAdapter(nn.Module):
    """Extract Boltz2 trunk single/pair representations without running diffusion.

    This module mirrors the trunk portion of Boltz2's forward pass, returning the
    single (s) and pair (z) representations needed for sequence-implied Potts
    generation and for structure-conditioning.
    """

    def __init__(
        self,
        input_embedder: InputEmbedder,
        rel_pos: RelativePositionEncoder,
        contact_conditioning: ContactConditioning,
        msa_module: MSAModule,
        pairformer_module: PairformerModule,
        s_init: nn.Linear,
        z_init_1: nn.Linear,
        z_init_2: nn.Linear,
        s_norm: nn.LayerNorm,
        z_norm: nn.LayerNorm,
        s_recycle: nn.Linear,
        z_recycle: nn.Linear,
        token_bonds: nn.Linear,
        token_bonds_type: Optional[nn.Embedding] = None,
        template_module: Optional[nn.Module] = None,
        use_templates: bool = False,
        bond_type_feature: bool = False,
        use_kernels: bool = False,
    ) -> None:
        super().__init__()
        self.input_embedder = input_embedder
        self.rel_pos = rel_pos
        self.contact_conditioning = contact_conditioning
        self.msa_module = msa_module
        self.pairformer_module = pairformer_module
        self.s_init = s_init
        self.z_init_1 = z_init_1
        self.z_init_2 = z_init_2
        self.s_norm = s_norm
        self.z_norm = z_norm
        self.s_recycle = s_recycle
        self.z_recycle = z_recycle
        self.token_bonds = token_bonds
        self.token_bonds_type = token_bonds_type
        self.template_module = template_module
        self.use_templates = use_templates
        self.bond_type_feature = bond_type_feature
        self.use_kernels = use_kernels

    @classmethod
    def from_boltz2_model(cls, model: nn.Module) -> "Boltz2TrunkAdapter":
        """Build an adapter from a Boltz2 model instance."""
        template_module = None
        use_templates = getattr(model, "use_templates", False)
        if use_templates:
            template_module = model.template_module
        return cls(
            input_embedder=model.input_embedder,
            rel_pos=model.rel_pos,
            contact_conditioning=model.contact_conditioning,
            msa_module=model.msa_module,
            pairformer_module=model.pairformer_module,
            s_init=model.s_init,
            z_init_1=model.z_init_1,
            z_init_2=model.z_init_2,
            s_norm=model.s_norm,
            z_norm=model.z_norm,
            s_recycle=model.s_recycle,
            z_recycle=model.z_recycle,
            token_bonds=model.token_bonds,
            token_bonds_type=getattr(model, "token_bonds_type", None),
            template_module=template_module,
            use_templates=use_templates,
            bond_type_feature=getattr(model, "bond_type_feature", False),
            use_kernels=getattr(model, "use_kernels", False),
        )

    def forward(self, feats: dict[str, Tensor], recycling_steps: int = 0) -> TrunkOutputs:
        s_inputs = self.input_embedder(feats)

        s_init = self.s_init(s_inputs)
        z_init = self.z_init_1(s_inputs)[:, :, None] + self.z_init_2(s_inputs)[:, None, :]
        z_init = z_init + self.rel_pos(feats)
        z_init = z_init + self.token_bonds(feats["token_bonds"].float())
        if self.bond_type_feature and self.token_bonds_type is not None:
            z_init = z_init + self.token_bonds_type(feats["type_bonds"].long())
        z_init = z_init + self.contact_conditioning(feats)

        s = torch.zeros_like(s_init)
        z = torch.zeros_like(z_init)
        mask = feats["token_pad_mask"].float()
        pair_mask = mask[:, :, None] * mask[:, None, :]

        for _ in range(recycling_steps + 1):
            s = s_init + self.s_recycle(self.s_norm(s))
            z = z_init + self.z_recycle(self.z_norm(z))
            if self.use_templates and self.template_module is not None:
                z = z + self.template_module(z, feats, pair_mask, use_kernels=self.use_kernels)
            z = z + self.msa_module(z, s_inputs, feats, use_kernels=self.use_kernels)
            s, z = self.pairformer_module(
                s, z, mask=mask, pair_mask=pair_mask, use_kernels=self.use_kernels
            )
            
        return TrunkOutputs(
            s_trunk=s,
            z_trunk=z,
            mask=mask,
            pair_mask=pair_mask,
            s_inputs=s_inputs,
            z_init=z_init,
        )


class SequencePottsHead(nn.Module):
    """Project Boltz2 pairwise embeddings into a dense Potts table."""

    def __init__(self, pair_dim: int, potts_dim: int = 400) -> None:
        super().__init__()
        self.proj = nn.Linear(pair_dim, potts_dim, bias=True)

    def forward(self, z_trunk: Tensor) -> Tensor:
        """Return dense Potts couplings with shape (B, N, N, potts_dim)."""
        return self.proj(z_trunk)


class PottsTableMixer(nn.Module):
    """Mix structure-derived and sequence-derived Potts tables."""

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        etab_geom: Tensor,
        e_idx: Tensor,
        etab_seq_dense: Tensor,
    ) -> Tensor:
        """Mix expanded geometry Potts with dense sequence Potts tables."""
        b, n, k, h = etab_geom.shape
        h2 = int(h**0.5)
        etab_geom_dense = expand_etab(etab_geom.view(b, n, k, h2, h2), e_idx)
        etab_geom_dense = etab_geom_dense.reshape(b, n, n, h)
        return self.alpha * etab_geom_dense + (1.0 - self.alpha) * etab_seq_dense
