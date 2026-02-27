from __future__ import annotations

import numpy as np

from model import CVAE
from utils import (
    collect_new_unique_from_raw,
    compose_train_config_from_dict,
    convert_to_smiles,
    infer_training_config_path,
    load_checkpoint_model_config,
    load_sampling_metadata,
    load_json,
    load_training_canonical_smiles,
    resolve_checkpoint_path,
)


def main() -> None:
    # Keep this aligned with sample.py defaults
    config = {
        "batch_size": 64,
        "save_file": None,
        "run_dir": "save/your_run_folder",
        "checkpoint_glob": "model_best.ckpt-*.pt",
        "training_config_file": None,
        "target_prop": "300.0 3.0",
        "seq_length": None,
        "mean": None,
        "stddev": None,
        "exclude_training": True,
        # diagnostics
        "batches": 200,
        "eos_token": "E",
    }

    config["save_file"] = resolve_checkpoint_path(
        save_file=config.get("save_file"),
        run_dir=config.get("run_dir"),
        checkpoint_glob=str(config.get("checkpoint_glob", "model_best.ckpt-*.pt")),
    )
    print(f"resolved checkpoint: {config['save_file']}")

    training_config_path = config["training_config_file"]

    model_config = load_checkpoint_model_config(config["save_file"])
    if model_config is not None:
        print("loaded model config from checkpoint metadata")
    else:
        if training_config_path is None:
            training_config_path = infer_training_config_path(config["save_file"])
        training_config = load_json(training_config_path)
        print(f"loaded training config from: {training_config_path}")
        model_config = compose_train_config_from_dict(training_config)
    for key in ["batch_size", "seq_length", "mean", "stddev"]:
        if config.get(key) is not None:
            model_config[key] = config[key]

    charset, vocab, inferred_num_prop = load_sampling_metadata(
        model_config["prop_file"],
        int(model_config["seq_length"]),
    )
    vocab_size = len(charset)
    # Print the resolved vocabulary size for quick sampling diagnostics.
    print(
        f"sampling metadata: vocab_size={vocab_size}, "
        f"num_prop={int(inferred_num_prop)}, prop_file={model_config['prop_file']}"
    )
    model_config["num_prop"] = int(inferred_num_prop)

    if bool(config.get("exclude_training", True)):
        training_smiles = load_training_canonical_smiles(model_config["prop_file"], int(model_config["seq_length"]))
        print(f"training molecules available for exclusion: {len(training_smiles)}")
    else:
        training_smiles = set()

    model = CVAE(vocab_size, model_config)
    model.restore(config["save_file"])

    target_row = [float(p) for p in str(config["target_prop"]).split()]
    target_prop = np.array([target_row for _ in range(int(model_config["batch_size"]))], dtype=np.float32)

    prop_norm_mean = model_config.get("prop_norm_mean")
    prop_norm_std = model_config.get("prop_norm_std")
    if prop_norm_mean is not None and prop_norm_std is not None:
        mean_arr = np.array(prop_norm_mean, dtype=np.float32)
        std_arr = np.array(prop_norm_std, dtype=np.float32)
        std_arr = np.where(std_arr < 1e-8, 1.0, std_arr)
        target_prop = (target_prop - mean_arr) / std_arr

    start_codon = np.array([np.array([vocab["X"]]) for _ in range(int(model_config["batch_size"]))])

    seen_smiles: set[str] = set()
    totals = {
        "total_generated": 0,
        "accepted": 0,
        "invalid_or_empty": 0,
        "discarded_cleanup": 0,
        "in_training": 0,
        "duplicate": 0,
    }

    for b in range(1, int(config["batches"]) + 1):
        latent_vector = np.random.normal(float(model_config["mean"]), float(model_config["stddev"]), (int(model_config["batch_size"]), int(model_config["latent_size"])))
        generated = model.sample(latent_vector, target_prop, start_codon, int(model_config["seq_length"]))
        raw = [convert_to_smiles(generated[i], charset) for i in range(len(generated))]

        accepted, stats = collect_new_unique_from_raw(
            raw_strings=raw,
            seen_smiles=seen_smiles,
            training_smiles=training_smiles,
            eos_token=str(config["eos_token"]),
        )
        for k in totals:
            totals[k] += int(stats.get(k, 0))

        if b == 1 or b % 10 == 0:
            print(
                f"batch={b:4d} accepted={stats['accepted']:3d} invalid={stats['invalid_or_empty']:3d} "
                f"discarded_cleanup={int(stats.get('discarded_cleanup', 0)):3d} "
                f"in_training={stats['in_training']:3d} dup={stats['duplicate']:3d} "
                f"total_unique_seen={len(seen_smiles)}"
            )

    print("\nTOTALS")
    print(totals)
    if totals["total_generated"]:
        acc = totals["accepted"] / totals["total_generated"]
        print(f"accepted share: {acc:.4%}")


if __name__ == "__main__":
    main()
