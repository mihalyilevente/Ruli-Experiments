"""Run the unchanged RULI helpers, with bounded smoke overrides kept explicit."""
from __future__ import annotations

import contextlib
import copy
import gc
import tempfile
from pathlib import Path
from types import SimpleNamespace

from common import (COMMON, REFERENCE, load_model, load_tokenizer, model_spec,
                    source_metadata, tokenizer_metadata, write_json)

EPOCHS = {"initial_sft": 5, "prefix": 1, "npo": 15, "final_retain_ft": 2}


def validate_stage_records(records, seed):
    if set(records) != set(EPOCHS):
        raise ValueError("Missing required training stages.")
    for stage, epochs in EPOCHS.items():
        record = records[stage]
        args = record["training_arguments"]
        sft = stage in ("initial_sft", "final_retain_ft")
        expected = {
            "num_train_epochs": epochs, "max_steps": -1, "seed": seed, "data_seed": seed,
            "per_device_train_batch_size": 4 if stage == "prefix" else 16,
            "gradient_accumulation_steps": 2 if stage == "npo" else 1,
            "learning_rate": 1e-5 if stage == "prefix" else 5e-5,
            "weight_decay": 0.01 if sft else 0.0,
            "load_best_model_at_end": sft,
            "adam_beta1": 0.9, "adam_beta2": 0.999, "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0, "warmup_steps": 0, "warmup_ratio": 0.0,
            "lr_scheduler_type": "linear", "optim": "adamw_torch",
            "fp16": False, "bf16": False,
        }
        if (record["smoke_only"] is not False or record["requested_epochs"] != epochs
                or record["global_step"] < 1 or any(args.get(k) != v for k, v in expected.items())):
            raise ValueError(f"Changed hyperparameters or smoke contamination: {stage}")
    if records["final_retain_ft"]["completed_epochs"] != 2:
        raise ValueError("Final retain FT must complete two epochs.")


@contextlib.contextmanager
def stage_settings(utils, seed, stage, smoke, device, record):
    """Scope constructor instrumentation; no persistent monkey-patch of upstream."""
    from transformers import TrainerCallback
    original_args, original_trainer, original_unlearner = (
        utils.TrainingArguments, utils.Trainer, utils.PrefixUnlearn)

    def arguments(*args, **kwargs):
        kwargs.update(seed=seed, data_seed=seed)
        if smoke:
            kwargs.update(max_steps=1, per_device_train_batch_size=1,
                          per_device_eval_batch_size=1, gradient_accumulation_steps=1,
                          save_total_limit=1, disable_tqdm=True, logging_steps=1)
        if str(device) == "cpu":
            kwargs["use_cpu"] = True
        result = original_args(*args, **kwargs)
        record["training_arguments"] = result.to_dict()
        return result

    class Recorder(TrainerCallback):
        def on_train_end(self, args, state, control, **kwargs):
            record["completed_epochs"] = state.epoch
            record["global_step"] = state.global_step
            record["best_model_checkpoint"] = state.best_model_checkpoint
            if not smoke and stage == "final_retain_ft" and state.epoch != 2:
                raise ValueError("Final retain fine-tuning did not complete exactly two epochs.")

    def observed(constructor):
        def build(*args, **kwargs):
            result = constructor(*args, **kwargs)
            result.add_callback(Recorder())
            return result
        return build

    utils.TrainingArguments = arguments
    utils.Trainer = observed(original_trainer)
    utils.PrefixUnlearn = observed(original_unlearner)
    try:
        yield
    finally:
        utils.TrainingArguments, utils.Trainer, utils.PrefixUnlearn = (
            original_args, original_trainer, original_unlearner)


def run_stages(model, tokenizer, initial, retain, forget, valid, utils, root,
               seed, device, smoke=False, measure=None):
    """Identical 2B stage order and hyperparameters; return explicit completion evidence."""
    import torch
    records = {}
    for stage, epochs in EPOCHS.items():
        print(f"[STAGE] {stage} ({'one smoke step' if smoke else str(epochs) + ' epochs'})", flush=True)
        record = {"requested_epochs": epochs, "smoke_only": smoke}
        before = COMMON._parameter_sha256(model, torch) if smoke else None
        with COMMON._working_directory(root / "trainer_work" / stage), stage_settings(
                utils, seed, stage, smoke, device, record):
            if stage in ("initial_sft", "final_retain_ft"):
                data = initial if stage == "initial_sft" else retain
                model = utils.train_sft(model, data, valid, tokenizer, epochs)
            elif stage == "prefix":
                model = utils.train_prefix(model, initial, valid, tokenizer, epochs)
            else:
                args = SimpleNamespace(unlearn_method="npo", unlearn_epochs=epochs, device=device)
                model = utils.unlearn_model(model, forget, retain, valid, tokenizer, args)
        if record.get("global_step", 0) < 1:
            raise ValueError(f"{stage} completed no optimizer steps.")
        if smoke:
            record["parameters_changed"] = before != COMMON._parameter_sha256(model, torch)
            if not record["parameters_changed"]:
                raise ValueError(f"{stage} failed to update model parameters.")
        for value in model.parameters():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite parameters after {stage}.")
        if stage in ("prefix", "final_retain_ft"):
            label = "pre_npo" if stage == "prefix" else "final"
            checkpoint = root / label
            COMMON._save_checkpoint(model, tokenizer, checkpoint)
            record["checkpoint"] = str(checkpoint.resolve())
            record["parameter_sha256"] = COMMON._parameter_sha256(model, torch)
            if measure:
                measure(label, model)
        records[stage] = record
        write_json(root / f"{stage}_completion.json", record)
    return model, records


def check_npo_objective(model, tokenizer, utils, device):
    """Exercise the real compute_loss, including backward, and verify its formula."""
    import torch
    batch = tokenizer(["A small reference sentence tests the unchanged unlearning objective."], return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    reference = copy.deepcopy(model).eval()
    # Avoid allocating another Trainer/optimizer just to test the actual loss method.
    trainer = object.__new__(utils.PrefixUnlearn)
    trainer.ref_model, trainer.beta, trainer.loss_type = reference, 0.1, "npo"
    trainer.state = SimpleNamespace(global_step=1)
    model.eval()
    forget, retain = copy.deepcopy(batch), copy.deepcopy(batch)
    loss, outputs = trainer.compute_loss(model, (forget, retain), return_outputs=True)
    with torch.no_grad():
        expected = -torch.nn.functional.logsigmoid(
            0.1 * (reference(**forget).logits - outputs.logits)).mean() * 2 / 0.1
    if not torch.allclose(loss, expected) or not torch.isfinite(loss):
        raise AssertionError("NPO no longer matches the upstream full-logit formula.")
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients) or not any(g.abs().sum() > 0 for g in gradients):
        raise AssertionError("NPO backward did not produce finite nonzero gradients.")
    value = float(loss.detach())
    model.zero_grad(set_to_none=True)
    del reference, trainer, outputs, loss
    return {"formula_unchanged": True, "finite_nonzero_gradients": True, "loss": value}


def smoke(name, ruli_root, device="cpu", report_path=None):
    import numpy as np
    import torch
    from datasets import Dataset
    utils = COMMON._load_ruli_utils(Path(ruli_root) / "text")
    COMMON._reset_rng(42, torch, np)
    tokenizer = load_tokenizer(name)
    model = load_model(name, device)
    metadata = {
        "model": model_spec(name), "architecture": type(model).__name__,
        "model_config": model.config.to_dict(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "tokenizer": tokenizer_metadata(tokenizer, name), "smoke_only": True,
        "research_result": False, "source_hashes": source_metadata(ruli_root),
        "package_versions": COMMON._software_metadata(torch),
        "overrides": "Two synthetic texts; one step per stage; batch size 1; gradient accumulation 1. No research data membership claim.",
    }
    metadata["npo_forward_backward"] = check_npo_objective(model, tokenizer, utils, device)
    encoded = tokenizer([
        "The quick brown fox jumps over the lazy dog beside a quiet river.",
        "A second short example checks causal language model training and evaluation."
    ])
    dataset = Dataset.from_dict({k: encoded[k] for k in ("input_ids", "attention_mask")})
    initial_hash = COMMON._parameter_sha256(model, torch)
    with tempfile.TemporaryDirectory(prefix="ruli_model_smoke_") as directory:
        model, records = run_stages(model, tokenizer, dataset, dataset, dataset, dataset,
                                   utils, Path(directory), 42, device, smoke=True)
        ids = [0, 1]
        tokens = {i: dataset[i]["input_ids"] for i in ids}
        losses = REFERENCE._run_reference_inference(model, dataset, tokenizer, device, utils, ids, tokens)
        final_hash = COMMON._parameter_sha256(model, torch)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = load_model(name, device, Path(directory) / "final")
        if COMMON._parameter_sha256(model, torch) != final_hash:
            raise ValueError("Save/reload changed parameter identity.")
        reloaded = REFERENCE._run_reference_inference(model, dataset, tokenizer, device, utils, ids, tokens)
        if not np.allclose(losses, reloaded, rtol=0, atol=1e-6):
            raise ValueError("Save/reload changed last-seven losses.")
        metadata.update(stages=records, initial_parameter_sha256=initial_hash,
                        final_parameter_sha256=final_hash, last_seven_losses=losses,
                        checkpoint_reload_passed=True,
                        reference_loss=REFERENCE._validate_reference_loss_path(utils, torch),
                        passed=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if report_path:
        write_json(report_path, metadata)
    return metadata
