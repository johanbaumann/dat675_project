from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np
from rdkit import RDLogger

from model import CVAE
from utils import (
    collect_new_unique_from_raw,
    compose_train_config_from_dict,
    convert_to_smiles,
    infer_training_config_path,
    load_sampling_metadata,
    load_json,
    load_training_canonical_smiles,
)


@dataclass(frozen=True)
class SweepSetting:
    temperature: float
    top_k: Optional[int]


def _fmt_top_k(value: Optional[int]) -> str:
    return "None" if value is None else str(int(value))


def run_setting(
    *,
    model: CVAE,
    charset: np.ndarray,
    vocab: dict,
    model_config: dict,
    target_prop: np.ndarray,
    training_smiles: set,
    batches: int,
    setting: SweepSetting,
    seed: int,
) -> dict:
    rng = np.random.default_rng(int(seed))
    batch_size = int(model_config["batch_size"])
    latent_size = int(model_config["latent_size"])
    seq_length = int(model_config["seq_length"])

    start_codon = np.array([np.array([vocab["X"]]) for _ in range(batch_size)])

    seen_smiles: set[str] = set()
    totals = {"total_generated": 0, "accepted": 0, "invalid_or_empty": 0, "in_training": 0, "duplicate": 0}

    for _ in range(int(batches)):
        latent_vector = rng.normal(
            float(model_config["mean"]),
            float(model_config["stddev"]),
            size=(batch_size, latent_size),
        )
        generated = model.sample(
            latent_vector,
            target_prop,
            start_codon,
            seq_length,
            do_sample=True,
            temperature=float(setting.temperature),
            top_k=setting.top_k,
        )
        raw = [convert_to_smiles(generated[i], charset) for i in range(len(generated))]

        _, stats = collect_new_unique_from_raw(
            raw_strings=raw,
            seen_smiles=seen_smiles,
            training_smiles=training_smiles,
            eos_token="E",
        )

        for k in totals:
            totals[k] += int(stats.get(k, 0))

    total = max(1, int(totals["total_generated"]))
    return {
        "temperature": float(setting.temperature),
        "top_k": setting.top_k,
        "samples": int(totals["total_generated"]),
        "accepted": int(totals["accepted"]),
        "unique_novel": int(totals["accepted"]),
        "invalid": int(totals["invalid_or_empty"]),
        "in_training": int(totals["in_training"]),
        "dup": int(totals["duplicate"]),
        "accepted_rate": float(totals["accepted"]) / float(total),
        "invalid_rate": float(totals["invalid_or_empty"]) / float(total),
        "dup_rate": float(totals["duplicate"]) / float(total),
        "in_training_rate": float(totals["in_training"]) / float(total),
    }


def main() -> None:
    # Silence RDKit SMILES parse spam during sweeps.
    RDLogger.logger().setLevel(logging.CRITICAL)

    # Align these with sample.py defaults
    save_file = "save/model_9.ckpt-9.pt"
    training_config_path = infer_training_config_path(save_file)
    training_config = load_json(training_config_path)
    model_config = compose_train_config_from_dict(training_config)

    charset, vocab, inferred_num_prop = load_sampling_metadata(
        model_config["prop_file"],
        int(model_config["seq_length"]),
    )
    model_config["num_prop"] = int(inferred_num_prop)

    vocab_size = len(charset)
    model = CVAE(vocab_size, model_config)
    model.restore(save_file)

    # Target property (MW, LogP)
    target_row = [300.0, 3.0]
    target_prop = np.array([target_row for _ in range(int(model_config["batch_size"]))], dtype=np.float32)

    # Apply property normalization if present
    prop_norm_mean = model_config.get("prop_norm_mean")
    prop_norm_std = model_config.get("prop_norm_std")
    if prop_norm_mean is not None and prop_norm_std is not None:
        mean_arr = np.array(prop_norm_mean, dtype=np.float32)
        std_arr = np.array(prop_norm_std, dtype=np.float32)
        std_arr = np.where(std_arr < 1e-8, 1.0, std_arr)
        target_prop = (target_prop - mean_arr) / std_arr

    training_smiles = load_training_canonical_smiles(model_config["prop_file"], int(model_config["seq_length"]))

    batches = 25  # 25 * 64 = 1600 samples per setting
    settings = [
        SweepSetting(temperature=t, top_k=k)
        for t in (0.6, 0.7, 0.8, 0.9, 1.0)
        for k in (10, 20, 50, 100)
    ]

    results: list[dict] = []
    start = time.time()
    for i, setting in enumerate(settings, start=1):
        r = run_setting(
            model=model,
            charset=charset,
            vocab=vocab,
            model_config=model_config,
            target_prop=target_prop,
            training_smiles=training_smiles,
            batches=batches,
            setting=setting,
            seed=1337 + i,
        )
        results.append(r)
        print(
            f"[{i:02d}/{len(settings)}] temp={setting.temperature:.1f} top_k={_fmt_top_k(setting.top_k):>4} "
            f"accepted={r['accepted']:4d}/{r['samples']} ({r['accepted_rate']:.2%}) invalid={r['invalid_rate']:.2%} dup={r['dup_rate']:.2%}"
        )

    # Sort by accepted_rate desc then invalid_rate asc
    results.sort(key=lambda x: (-x["accepted_rate"], x["invalid_rate"], x["dup_rate"]))

    print("\nTop 5 settings by novel+unique acceptance rate")
    for r in results[:5]:
        print(
            f"temp={r['temperature']:.1f} top_k={_fmt_top_k(r['top_k']):>4} "
            f"accepted_rate={r['accepted_rate']:.2%} invalid_rate={r['invalid_rate']:.2%} dup_rate={r['dup_rate']:.2%}"
        )

    # Emit a markdown table to stdout (copy/paste into README).
    print("\nMARKDOWN_TABLE")
    print(
        "| temperature | top_k | samples | accepted (unique+novel) | accepted_rate | invalid_rate | dup_rate | in_training_rate |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in results:
        print(
            f"| {r['temperature']:.1f} | {_fmt_top_k(r['top_k'])} | {r['samples']} | {r['accepted']} | "
            f"{r['accepted_rate']:.2%} | {r['invalid_rate']:.2%} | {r['dup_rate']:.2%} | {r['in_training_rate']:.2%} |"
        )

    dur = time.time() - start
    print(f"\nSweep duration: {dur:.1f}s")


if __name__ == "__main__":
    main()
