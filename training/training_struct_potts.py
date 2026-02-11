import argparse
import os.path


def alpha_schedule(epoch, step, start, end, warmup_epochs):
    if warmup_epochs <= 0:
        return end
    progress = min(epoch / max(warmup_epochs, 1), 1.0)
    return start + (end - start) * progress


def decoy_real_fraction_schedule(step, start_step, end_step, max_fraction):
    if max_fraction <= 0.0:
        return 0.0
    if end_step <= start_step:
        return float(max_fraction)
    progress = (step - start_step) / max(end_step - start_step, 1)
    progress = min(max(progress, 0.0), 1.0)
    return float(max_fraction) * progress


def load_boltz2_checkpoint(checkpoint_path, device):
    import torch

    from boltz.model.models.boltz2 import Boltz2

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})

    diffusion_process_args = hparams.get("diffusion_process_args")
    if not isinstance(diffusion_process_args, dict):
        return Boltz2.load_from_checkpoint(checkpoint_path, map_location=device)

    import inspect

    from boltz.model.modules.diffusionv2 import AtomDiffusion

    accepted_diffusion_keys = {
        name
        for name in inspect.signature(AtomDiffusion.__init__).parameters
        if name not in {"self", "score_model_args"}
    }
    filtered_diffusion_args = {
        key: value
        for key, value in diffusion_process_args.items()
        if key in accepted_diffusion_keys
    }

    return Boltz2.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        diffusion_process_args=filtered_diffusion_args,
    )


def get_boltz2_feats(batch):
    boltz2_feats = batch[0].get("boltz2_feats")
    if boltz2_feats is None:
        raise KeyError(
            "Expected batch[0]['boltz2_feats'] with Boltz2 feature dict."
        )
    return boltz2_feats


def validate_boltz2_alignment(boltz2_feats, seq_mask):
    token_mask = boltz2_feats.get("token_pad_mask")
    msa = boltz2_feats.get("msa")
    if token_mask is None:
        raise KeyError("Boltz2 features missing token_pad_mask.")
    if token_mask.shape[0] != seq_mask.shape[0]:
        raise ValueError(
            f"Boltz2 batch size {token_mask.shape[0]} does not match "
            f"sequence mask batch size {seq_mask.shape[0]}."
        )
    if token_mask.shape[1] != seq_mask.shape[1]:
        raise ValueError(
            f"Boltz2 token length {token_mask.shape[1]} does not match "
            f"sequence length {seq_mask.shape[1]}."
        )
    if msa is not None and msa.shape[-1] != seq_mask.shape[1]:
        raise ValueError(
            f"Boltz2 MSA length {msa.shape[-1]} does not match "
            f"sequence length {seq_mask.shape[1]}."
        )


def load_esm_model(model_name, device):
    import esm

    if model_name.startswith("esmc"):
        try:
            from esm.models.esmc import ESMC
        except Exception as exc:  # pragma: no cover - import guard
            raise ImportError(
                "ESM-C models require the evolutionaryscale/esm package."
            ) from exc

        esm_model = ESMC.from_pretrained(model_name)
        esm_model = esm_model.to(device)
        esm_model.eval()
        for param in esm_model.parameters():
            param.requires_grad = False

        esm_tokenizer = getattr(esm_model, "tokenizer", None)
        if esm_tokenizer is None:
            raise RuntimeError(
                "Unable to locate an ESM-C tokenizer for token mapping."
            )
        return esm_model, esm_tokenizer

    esm_model, esm_alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    esm_model = esm_model.to(device)
    esm_model.eval()
    for param in esm_model.parameters():
        param.requires_grad = False
    return esm_model, esm_alphabet


def _extract_encoded_tokens(encoded):
    if isinstance(encoded, dict):
        for key in ("input_ids", "tokens", "token_ids"):
            if key in encoded:
                encoded = encoded[key]
                break
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, (list, tuple)):
        return [int(token) for token in encoded]
    return []


def build_esm_token_map(esm_vocab, model_alphabet):
    import torch

    if hasattr(esm_vocab, "get_idx") or hasattr(esm_vocab, "tok_to_idx"):
        get_idx = getattr(esm_vocab, "get_idx", None)
        if get_idx is None:
            get_idx = esm_vocab.tok_to_idx.get
        unknown_idx = getattr(esm_vocab, "unk_idx", None)
        if unknown_idx is None:
            unknown_idx = get_idx("<unk>") if get_idx else 0
        token_map = []
        for aa in model_alphabet:
            try:
                token_map.append(get_idx(aa))
            except Exception:
                token_map.append(unknown_idx)
        return torch.tensor(token_map, dtype=torch.long)

    encode = getattr(esm_vocab, "encode", None)
    if encode is None:
        raise ValueError(
            "ESM tokenizer does not expose get_idx/tok_to_idx or encode."
        )

    special_ids = set()
    for attr in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "mask_token_id",
        "unk_token_id",
    ):
        token_id = getattr(esm_vocab, attr, None)
        if token_id is not None:
            special_ids.add(int(token_id))
    unknown_idx = getattr(esm_vocab, "unk_token_id", 0)
    token_map = []
    for aa in model_alphabet:
        encoded = _extract_encoded_tokens(encode(aa))
        if not encoded:
            token_map.append(int(unknown_idx))
            continue
        token_id = next((t for t in encoded if t not in special_ids), encoded[0])
        token_map.append(int(token_id))
    return torch.tensor(token_map, dtype=torch.long)


def main(args):
    import time
    import os
    import numpy as np
    import torch
    from torch import optim
    from concurrent.futures import ProcessPoolExecutor
    from utils import (
        worker_init_fn,
        get_pdbs,
        loader_pdb,
        build_training_clusters,
        PDB_dataset,
        StructureDataset,
        StructureLoader,
    )
    from model_utils_struct import (
        ProteinMPNN,
        featurize,
        loss_smoothed,
        loss_nll,
        get_std_opt,
    )
    from boltz2_adapter import Boltz2TrunkAdapter, SequencePottsHead
    from struct_potts_losses import (
        ESMCDecoySequencePool,
        potts_consistency_loss,
        msa_similarity_loss,
        msa_similarity_loss_esm,
        msa_similarity_loss_esmc,
        structure_consistency_loss,
        structure_fape_loss,
    )

    scaler = torch.cuda.amp.GradScaler()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Time-stamp outputs so multiple runs do not overwrite each other.
    base_folder = time.strftime(args.path_for_outputs, time.localtime())
    if base_folder[-1] != "/":
        base_folder += "/"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    subfolders = ["model_weights"]
    for subfolder in subfolders:
        if not os.path.exists(base_folder + subfolder):
            os.makedirs(base_folder + subfolder)

    logfile = base_folder + "log.txt"
    if not args.previous_checkpoint:
        with open(logfile, "w") as f:
            f.write("Epoch\tTrain\tValidation\n")

    data_path = args.path_for_training_data
    params = {
        "LIST": f"{data_path}/list.csv",
        "VAL": f"{data_path}/valid_clusters.txt",
        "TEST": f"{data_path}/test_clusters.txt",
        "DIR": f"{data_path}",
        "DATCUT": "2030-Jan-01",
        "RESCUT": args.rescut,
        "HOMO": 0.70,
    }

    load_param = {
        "batch_size": 1,
        "shuffle": True,
        "pin_memory": False,
        "num_workers": 4,
    }

    if args.debug:
        args.num_examples_per_epoch = 50
        args.max_protein_length = 1000
        args.batch_size = 1000

    train, valid, _ = build_training_clusters(params, args.debug)
    train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
    train_loader = torch.utils.data.DataLoader(
        train_set, worker_init_fn=worker_init_fn, **load_param
    )
    valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
    valid_loader = torch.utils.data.DataLoader(
        valid_set, worker_init_fn=worker_init_fn, **load_param
    )

    # ProteinMPNN acts as the sequence/structure head for Potts-style training.
    model = ProteinMPNN(
        node_features=args.hidden_dim,
        edge_features=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        k_neighbors=args.num_neighbors,
        dropout=args.dropout,
        augment_eps=args.backbone_noise,
        use_potts=True,
        struct_predict=True,
        struct_use_decoder_one_hot=True,
        struct_trunk_backend=args.struct_trunk_backend,
    )
    model.to(device)

    # Boltz2 provides the trunk representation and MSA features for Potts loss.
    boltz2_model = load_boltz2_checkpoint(args.boltz2_checkpoint, device)
    boltz2_model.to(device)
    boltz2_model.eval()
    boltz2_trunk = Boltz2TrunkAdapter.from_boltz2_model(boltz2_model)
    boltz2_trunk.to(device)
    seq_potts_head = SequencePottsHead(
        pair_dim=boltz2_model.hparams.token_z, potts_dim=args.potts_dim
    ).to(device)

    esm_model = None
    esm_token_map = None
    # Optional: add an ESM-based contrastive loss over MSA sequences.
    if args.msa_similarity_loss_type == "esm":
        esm_model, esm_vocab = load_esm_model(args.esm_model_name, device)
        # ProteinMPNN uses this alphabet ordering for token IDs.
        model_alphabet = "ACDEFGHIKLMNPQRSTVWYX-"
        esm_token_map = build_esm_token_map(esm_vocab, model_alphabet)
        esm_is_esmc = args.esm_model_name.startswith("esmc")
    else:
        esm_is_esmc = False

    esmc_decoy_pool = None
    if esm_is_esmc:
        esmc_decoy_pool = ESMCDecoySequencePool(max_size=args.esmc_decoy_pool_size)

    optimizer = get_std_opt(model, args.hidden_dim, args.warmup_steps)

    total_step = 0
    reload_c = 0
    with ProcessPoolExecutor(max_workers=2) as executor:
        q = executor.submit(
            get_pdbs,
            train_loader,
            1,
            args.max_protein_length,
            args.num_examples_per_epoch,
        )
        p = executor.submit(
            get_pdbs,
            valid_loader,
            1,
            args.max_protein_length,
            args.num_examples_per_epoch,
        )

        for epoch in range(args.num_epochs):
            t0 = time.time()
            model.train()
            train_sum, train_weights = 0.0, 0.0
            train_acc = 0.0
            if epoch % args.reload_data_every_n_epochs == 0:
                if reload_c != 0:
                    pdb_dict_train = q.get().result()
                    dataset_train = StructureDataset(
                        pdb_dict_train,
                        truncate=None,
                        max_length=args.max_protein_length,
                    )
                    loader_train = StructureLoader(
                        dataset_train, batch_size=args.batch_size
                    )
                    pdb_dict_valid = p.get().result()
                    dataset_valid = StructureDataset(
                        pdb_dict_valid,
                        truncate=None,
                        max_length=args.max_protein_length,
                    )
                    loader_valid = StructureLoader(
                        dataset_valid, batch_size=args.batch_size
                    )
                    q = executor.submit(
                        get_pdbs,
                        train_loader,
                        1,
                        args.max_protein_length,
                        args.num_examples_per_epoch,
                    )
                    p = executor.submit(
                        get_pdbs,
                        valid_loader,
                        1,
                        args.max_protein_length,
                        args.num_examples_per_epoch,
                    )
                reload_c += 1

            for _, batch in enumerate(loader_train):
                (
                    X,
                    S,
                    _,
                    mask,
                    lengths,
                    chain_M,
                    residue_idx,
                    mask_self,
                    chain_encoding_all,
                    _,
                    backbone_4x4,
                    _,
                ) = featurize(
                    batch,
                    device,
                    augment_type="atomic",
                    augment_eps=args.backbone_noise,
                    replicate=1,
                    epoch=epoch,
                    openfold_backbone=args.structure_loss_type == "fape",
                )
                backbone_4x4 = backbone_4x4.to(device)
                mask_for_loss = mask * chain_M
                boltz2_feats = get_boltz2_feats(batch)
                boltz2_feats = {
                    k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in boltz2_feats.items()
                }
                validate_boltz2_alignment(boltz2_feats, mask)

                optimizer.zero_grad()
                potts_alpha = alpha_schedule(
                    epoch, total_step, args.alpha_start, args.alpha_end, args.alpha_warmup_epochs
                )
                decoy_real_fraction = decoy_real_fraction_schedule(
                    total_step,
                    args.esmc_decoy_real_start_step,
                    args.esmc_decoy_real_end_step,
                    args.esmc_decoy_real_max_fraction,
                )

                if args.mixed_precision:
                    with torch.cuda.amp.autocast():
                        trunk_out = boltz2_trunk(boltz2_feats, args.boltz2_recycles)
                        etab_seq_dense = seq_potts_head(trunk_out.z_trunk)
                        log_probs, etab_geom, e_idx, frames, positions, logits = model(
                            X,
                            S,
                            mask,
                            chain_M,
                            residue_idx,
                            chain_encoding_all,
                            return_logits=True,
                            struct_etab_seq_dense=etab_seq_dense,
                            struct_potts_alpha=potts_alpha,
                        )
                        loss_potts = potts_consistency_loss(
                            etab_geom, e_idx, etab_seq_dense, mask
                        )
                        if args.msa_similarity_loss_type == "esm":
                            if esm_is_esmc:
                                loss_msa = msa_similarity_loss_esmc(
                                    log_probs,
                                    boltz2_feats["msa"],
                                    boltz2_feats["msa_mask"],
                                    mask,
                                    esm_model,
                                    esm_token_map,
                                    margin=args.msa_margin,
                                    gumbel_temperature=args.esm_gumbel_temperature,
                                    decoy_real_fraction=decoy_real_fraction,
                                    decoy_pool=esmc_decoy_pool,
                                )
                            else:
                                loss_msa = msa_similarity_loss_esm(
                                    log_probs,
                                    boltz2_feats["msa"],
                                    boltz2_feats["msa_mask"],
                                    mask,
                                    esm_model,
                                    esm_token_map,
                                    margin=args.msa_margin,
                                    gumbel_temperature=args.esm_gumbel_temperature,
                                    pred_embed_mode=args.esm_pred_embed_mode,
                                )
                        else:
                            loss_msa = msa_similarity_loss(
                                log_probs,
                                boltz2_feats["msa"],
                                boltz2_feats["msa_mask"],
                                mask,
                                margin=args.msa_margin,
                            )
                        if args.structure_loss_type == "fape":
                            loss_struct = structure_fape_loss(
                                frames, backbone_4x4, mask
                            )
                        else:
                            loss_struct = structure_consistency_loss(
                                positions, X, mask
                            )
                        loss_total = (
                            args.loss_potts_weight * loss_potts
                            + args.loss_msa_weight * loss_msa
                            + args.loss_struct_weight * loss_struct
                        )
                        _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                        loss_total = loss_total + args.loss_nll_weight * loss_av_smoothed
                    scaler.scale(loss_total).backward()
                    if args.gradient_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.gradient_norm
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    trunk_out = boltz2_trunk(boltz2_feats, args.boltz2_recycles)
                    etab_seq_dense = seq_potts_head(trunk_out.z_trunk)
                    log_probs, etab_geom, e_idx, frames, positions, logits = model(
                        X,
                        S,
                        mask,
                        chain_M,
                        residue_idx,
                        chain_encoding_all,
                        return_logits=True,
                        struct_etab_seq_dense=etab_seq_dense,
                        struct_potts_alpha=potts_alpha,
                    )
                    loss_potts = potts_consistency_loss(
                        etab_geom, e_idx, etab_seq_dense, mask
                    )
                    if args.msa_similarity_loss_type == "esm":
                        if esm_is_esmc:
                            loss_msa = msa_similarity_loss_esmc(
                                log_probs,
                                boltz2_feats["msa"],
                                boltz2_feats["msa_mask"],
                                mask,
                                esm_model,
                                esm_token_map,
                                margin=args.msa_margin,
                                gumbel_temperature=args.esm_gumbel_temperature,
                                decoy_real_fraction=decoy_real_fraction,
                                decoy_pool=esmc_decoy_pool,
                            )
                        else:
                            loss_msa = msa_similarity_loss_esm(
                                log_probs,
                                boltz2_feats["msa"],
                                boltz2_feats["msa_mask"],
                                mask,
                                esm_model,
                                esm_token_map,
                                margin=args.msa_margin,
                                gumbel_temperature=args.esm_gumbel_temperature,
                                pred_embed_mode=args.esm_pred_embed_mode,
                            )
                    else:
                        loss_msa = msa_similarity_loss(
                            log_probs,
                            boltz2_feats["msa"],
                            boltz2_feats["msa_mask"],
                            mask,
                            margin=args.msa_margin,
                        )
                    if args.structure_loss_type == "fape":
                        loss_struct = structure_fape_loss(
                            frames, backbone_4x4, mask
                        )
                    else:
                        loss_struct = structure_consistency_loss(positions, X, mask)
                    loss_total = (
                        args.loss_potts_weight * loss_potts
                        + args.loss_msa_weight * loss_msa
                        + args.loss_struct_weight * loss_struct
                    )
                    _, loss_av_smoothed = loss_smoothed(S, log_probs, mask_for_loss)
                    loss_total = loss_total + args.loss_nll_weight * loss_av_smoothed
                    loss_total.backward()
                    if args.gradient_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.gradient_norm
                        )
                    optimizer.step()

                loss, _, true_false = loss_nll(S, log_probs, mask_for_loss)
                train_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                train_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                train_weights += torch.sum(mask_for_loss).cpu().data.numpy()
                total_step += 1

            model.eval()
            with torch.no_grad():
                validation_sum, validation_weights = 0.0, 0.0
                validation_acc = 0.0
                potts_alpha = alpha_schedule(
                    epoch, total_step, args.alpha_start, args.alpha_end, args.alpha_warmup_epochs
                )
                for _, batch in enumerate(loader_valid):
                    (
                        X,
                        S,
                        _,
                        mask,
                        lengths,
                        chain_M,
                        residue_idx,
                        mask_self,
                        chain_encoding_all,
                        _,
                        backbone_4x4,
                        _,
                    ) = featurize(
                        batch,
                        device,
                        augment_type="atomic",
                        augment_eps=args.backbone_noise,
                        replicate=1,
                        epoch=epoch,
                        openfold_backbone=args.structure_loss_type == "fape",
                    )
                    backbone_4x4 = backbone_4x4.to(device)
                    boltz2_feats = get_boltz2_feats(batch)
                    boltz2_feats = {
                        k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in boltz2_feats.items()
                    }
                    validate_boltz2_alignment(boltz2_feats, mask)
                    trunk_out = boltz2_trunk(boltz2_feats, args.boltz2_recycles)
                    etab_seq_dense = seq_potts_head(trunk_out.z_trunk)
                    log_probs, etab_geom, e_idx, frames, positions, logits = model(
                        X,
                        S,
                        mask,
                        chain_M,
                        residue_idx,
                        chain_encoding_all,
                        return_logits=True,
                        struct_etab_seq_dense=etab_seq_dense,
                        struct_potts_alpha=potts_alpha,
                    )
                    mask_for_loss = mask * chain_M
                    loss, _, true_false = loss_nll(S, log_probs, mask_for_loss)
                    validation_sum += torch.sum(loss * mask_for_loss).cpu().data.numpy()
                    validation_acc += torch.sum(true_false * mask_for_loss).cpu().data.numpy()
                    validation_weights += torch.sum(mask_for_loss).cpu().data.numpy()

            train_loss = train_sum / train_weights
            train_accuracy = train_acc / train_weights
            train_perplexity = np.exp(train_loss)
            validation_loss = validation_sum / validation_weights
            validation_accuracy = validation_acc / validation_weights
            validation_perplexity = np.exp(validation_loss)

            train_perplexity_ = np.format_float_positional(
                np.float32(train_perplexity), unique=False, precision=3
            )
            validation_perplexity_ = np.format_float_positional(
                np.float32(validation_perplexity), unique=False, precision=3
            )
            train_accuracy_ = np.format_float_positional(
                np.float32(train_accuracy), unique=False, precision=3
            )
            validation_accuracy_ = np.format_float_positional(
                np.float32(validation_accuracy), unique=False, precision=3
            )

            t1 = time.time()
            dt = np.format_float_positional(np.float32(t1 - t0), unique=False, precision=1)
            with open(logfile, "a") as f:
                f.write(
                    f"epoch: {epoch+1}, step: {total_step}, time: {dt}, "
                    f"train: {train_perplexity_}, valid: {validation_perplexity_}, "
                    f"train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}\n"
                )
            print(
                f"epoch: {epoch+1}, step: {total_step}, time: {dt}, "
                f"train: {train_perplexity_}, valid: {validation_perplexity_}, "
                f"train_acc: {train_accuracy_}, valid_acc: {validation_accuracy_}"
            )

            checkpoint_filename_last = (
                base_folder + "model_weights/epoch_last.pt"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": total_step,
                    "num_edges": args.num_neighbors,
                    "noise_level": args.backbone_noise,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.optimizer.state_dict(),
                },
                checkpoint_filename_last,
            )

            if (epoch + 1) % args.save_model_every_n_epochs == 0:
                checkpoint_filename = (
                    base_folder + f"model_weights/epoch{epoch+1}_step{total_step}.pt"
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "step": total_step,
                        "num_edges": args.num_neighbors,
                        "noise_level": args.backbone_noise,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.optimizer.state_dict(),
                    },
                    checkpoint_filename,
                )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--path_for_training_data", type=str, default="my_path/pdb_2021aug02")
    argparser.add_argument("--path_for_outputs", type=str, default="./test")
    argparser.add_argument("--previous_checkpoint", type=str, default="")
    argparser.add_argument("--num_epochs", type=int, default=200)
    argparser.add_argument("--save_model_every_n_epochs", type=int, default=10)
    argparser.add_argument("--reload_data_every_n_epochs", type=int, default=2)
    argparser.add_argument("--num_examples_per_epoch", type=int, default=1000000)
    argparser.add_argument("--batch_size", type=int, default=10000)
    argparser.add_argument("--max_protein_length", type=int, default=10000)
    argparser.add_argument("--hidden_dim", type=int, default=128)
    argparser.add_argument("--num_encoder_layers", type=int, default=3)
    argparser.add_argument("--num_decoder_layers", type=int, default=3)
    argparser.add_argument("--num_neighbors", type=int, default=48)
    argparser.add_argument("--dropout", type=float, default=0.1)
    argparser.add_argument("--backbone_noise", type=float, default=0.2)
    argparser.add_argument("--rescut", type=float, default=3.5)
    argparser.add_argument("--debug", type=bool, default=False)
    argparser.add_argument("--gradient_norm", type=float, default=-1.0)
    argparser.add_argument("--mixed_precision", type=bool, default=True)
    argparser.add_argument("--warmup_steps", type=int, default=4000)

    argparser.add_argument("--boltz2_checkpoint", type=str, required=True)
    argparser.add_argument("--boltz2_recycles", type=int, default=1)

    argparser.add_argument(
        "--struct_trunk_backend",
        type=str,
        default="boltz2",
        choices=["boltz2"],
        help="Structure trunk backend used inside ProteinMPNN's folding module.",
    )
    argparser.add_argument("--potts_dim", type=int, default=400)

    argparser.add_argument("--alpha_start", type=float, default=1.0)
    argparser.add_argument("--alpha_end", type=float, default=0.0)
    argparser.add_argument("--alpha_warmup_epochs", type=int, default=100)

    argparser.add_argument("--loss_potts_weight", type=float, default=1.0)
    argparser.add_argument("--loss_msa_weight", type=float, default=1.0)
    argparser.add_argument("--loss_struct_weight", type=float, default=1.0)
    argparser.add_argument("--loss_nll_weight", type=float, default=1.0)
    argparser.add_argument("--msa_margin", type=float, default=0.1)
    argparser.add_argument(
        "--msa_similarity_loss_type",
        type=str,
        default="log_probs",
        choices=["log_probs", "esm"],
    )
    argparser.add_argument(
        "--esm_model_name",
        type=str,
        default="esmc_300m",
    )
    argparser.add_argument(
        "--esm_gumbel_temperature",
        type=float,
        default=1.0,
    )
    argparser.add_argument(
        "--esm_pred_embed_mode",
        type=str,
        default="gumbel_st",
        choices=["gumbel_st", "potts_weighted"],
    )
    argparser.add_argument(
        "--esmc_decoy_real_start_step",
        type=int,
        default=0,
        help="Training step to start mixing real-sequence decoys into ESM-C loss.",
    )
    argparser.add_argument(
        "--esmc_decoy_real_end_step",
        type=int,
        default=20000,
        help="Training step where real-sequence decoy fraction reaches max.",
    )
    argparser.add_argument(
        "--esmc_decoy_real_max_fraction",
        type=float,
        default=1.0,
        help="Maximum fraction of ESM-C decoys drawn from real-sequence pool.",
    )
    argparser.add_argument(
        "--esmc_decoy_pool_size",
        type=int,
        default=8192,
        help="Max number of cached real decoy sequences per sequence length.",
    )
    argparser.add_argument(
        "--structure_loss_type",
        type=str,
        default="ca",
        choices=["ca", "fape"],
    )

    main(argparser.parse_args())
