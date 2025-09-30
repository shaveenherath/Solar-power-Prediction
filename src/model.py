# model.py
from __future__ import annotations
import os, json, copy, time
from typing import Dict, Any, List, Tuple
import numpy as np
import torch

from src.utils.data_loading import create_dataloaders_grouped
from src.utils.training import train_model, evaluate_model
from src.models import LSTMForecast, TimeSeriesTransformer, CNNTransformerHybrid, LLaMATimeSeries , CNNTimeseries , CNNLSTMTimeSeries , BiLSTMTimeSeries


# -------------------------
# Registry for model classes
# -------------------------
MODEL_REGISTRY = {
    "lstm": LSTMForecast,
    "transformer": TimeSeriesTransformer,
    "cnn_transformer": CNNTransformerHybrid,
    "llama_ts": LLaMATimeSeries,
    "cnn": CNNTimeseries,
    "cnn_lstm": CNNLSTMTimeSeries,
    "bilstm": BiLSTMTimeSeries,
}



def _ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)


def _seed_everything(seed: int | None):
    if seed is None:
        return
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _select_device(pref: str = "cuda") -> str:
    return pref if (pref == "cuda" and torch.cuda.is_available()) else "cpu"


def _build_dataloaders(cfg: Dict[str, Any]):
    dl = cfg["dataloaders"]
    te = cfg["time_encoding"]
    lf = cfg["lag_features"]
    return create_dataloaders_grouped(
        csv_file=dl["csv_file"],
        feature_cols=dl["feature_cols"],
        target_col=dl.get("target_col", "y_norm"),
        lookback=dl.get("lookback", 24),
        horizon=dl.get("horizon", 1),
        batch_size=dl.get("batch_size", 128),
        train_frac=dl.get("train_frac", 0.7),
        val_frac=dl.get("val_frac", 0.2),
        num_workers=dl.get("num_workers", 0),
        pin_memory=dl.get("pin_memory", False),
        shuffle_train=dl.get("shuffle_train", True),
        time_encoding=te ,
        lag_cfg=lf 

    )


def _init_model(model_name: str, input_size: int, model_cfg: dict) -> torch.nn.Module:
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")

    # LSTM
    if model_name == "lstm":
        return MODEL_REGISTRY[model_name](
            input_size=input_size,
            hidden_size=model_cfg.get("hidden_size", 128),
            num_layers=model_cfg.get("num_layers", 2),
            dropout=model_cfg.get("dropout", 0.2),
        )

    # Transformer
    if model_name == "transformer":
        return MODEL_REGISTRY[model_name](
            input_size=input_size,
            num_heads=model_cfg.get("num_heads", max(1, input_size // 2)),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layers=model_cfg.get("num_layers", 6),
            dropout=model_cfg.get("dropout", 0.1),
        )

    # CNN + Transformer Hybrid
    if model_name == "cnn_transformer":
        return MODEL_REGISTRY[model_name](
            input_size=input_size,
            cnn_out_channels=model_cfg.get("cnn_out_channels", 64),
            kernel_size=model_cfg.get("kernel_size", 3),
            num_heads=model_cfg.get("num_heads", 8),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            num_layers=model_cfg.get("num_layers", 6),
            dropout=model_cfg.get("dropout", 0.1),
        )

    # LLaMA Time Series
    if model_name == "llama_ts":
        return MODEL_REGISTRY[model_name](
            n_features=input_size,
            d_model=model_cfg.get("d_model", 128),
            n_heads=model_cfg.get("n_heads", 8),
            n_layers=model_cfg.get("n_layers", 4),
            dropout=model_cfg.get("dropout", 0.1),
            max_seq_len=model_cfg.get("max_seq_len", 48),
        )

    # Pure CNN
    if model_name == "cnn":
        return MODEL_REGISTRY[model_name](
            input_size=input_size,
            num_filters=model_cfg.get("num_filters", 64),
            kernel_size=model_cfg.get("kernel_size", 3),
            dropout=model_cfg.get("dropout", 0.1),
        )

    # CNN + LSTM
    if model_name == "cnn_lstm":
        return MODEL_REGISTRY[model_name](
            input_size=input_size,
            hidden_size=model_cfg.get("hidden_size", 64),
            num_layers=model_cfg.get("num_layers", 1),
            cnn_filters=model_cfg.get("cnn_filters", 32),
            kernel_size=model_cfg.get("kernel_size", 3),
            dropout=model_cfg.get("dropout", 0.1),
        )

    # Bidirectional LSTM
    if model_name == "bilstm":
        return MODEL_REGISTRY[model_name](
            input_size=input_size,
            hidden_size=model_cfg.get("hidden_size", 64),
            num_layers=model_cfg.get("num_layers", 1),
            dropout=model_cfg.get("dropout", 0.1),
        )

    # Fallback
    raise RuntimeError(f"Failed to initialize model '{model_name}'")



def run_single_model(cfg: Dict[str, Any], model_key: str) -> Dict[str, Any]:
    """
    Train + evaluate a single model specified by 'model_key' in cfg['models'].
    Returns a dict of results (MAE/RMSE, paths, etc.)
    """
    _seed_everything(cfg.get("seed"))
    device = _select_device(cfg.get("device", "cuda"))

    # --- data ---
    train_loader, val_loader, test_loader, feature_cols, scaler, (idx_tr, idx_va, idx_te) = _build_dataloaders(cfg)
    input_size = len(feature_cols)
    

    # --- model + train settings ---
    model_cfg_all = cfg.get("models", {})
    if model_key not in model_cfg_all:
        raise ValueError(f"Model '{model_key}' not found in config.models")
    model_cfg = model_cfg_all[model_key] or {}
    model_name = model_cfg.get("name", model_key)

    model = _init_model(model_key, input_size, model_cfg)

    # results dir per model
    results_base = cfg.get("results_base", "results")
    model_dir = os.path.join(results_base, model_name)
    _ensure_dirs(model_dir)

    # --- training ---
    train_cfg = cfg.get("training", {})
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        model_name=model_name,
        results_base=results_base,
        epochs=train_cfg.get("epochs", 50),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        device=device,
        hyperparams={
            "lookback": cfg["dataloaders"].get("lookback", 24),
            "horizon": cfg["dataloaders"].get("horizon", 1),
            "feature_cols": cfg["dataloaders"]["feature_cols"],
            **{k: v for k, v in model_cfg.items() if k not in ("name",)},
        },
        early_stop_patience=train_cfg.get("early_stop_patience", 15),
        early_stop_min_delta=train_cfg.get("early_stop_min_delta", 0.0),
    )

    # --- evaluation (test) ---
    mae, rmse,r2, preds, trues = evaluate_model(model, test_loader, device=device, denorm=None)

    # Save eval summary
    summary = {
        "model": model_name,
        "device": device,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2" : float(r2),
        "n_test": int(len(trues)),
        "timestamp": int(time.time()),
    }
    with open(os.path.join(model_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Optionally save raw preds/trues
    if cfg.get("save_predictions", False):
        np.save(os.path.join(model_dir, "preds.npy"), preds)
        np.save(os.path.join(model_dir, "trues.npy"), trues)

    print(f"[{model_name}] Test MAE={summary['mae']:.4f} | RMSE={summary['rmse']:.4f}")
    return summary


def run(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run training/eval for:
      - a single model (cfg['run']['model'] = 'lstm'), or
      - all listed models (cfg['run']['model'] = 'all')
    Returns dict of summaries keyed by model key.
    """
    # Ensure base dirs
    _ensure_dirs(cfg.get("results_base", "results"))

    which = cfg.get("run", {}).get("model", "lstm")
    summaries: Dict[str, Any] = {}

    if which == "all":
        for model_key in cfg.get("models", {}).keys():
            summaries[model_key] = run_single_model(cfg, model_key)
    else:
        summaries[which] = run_single_model(cfg, which)

    # Save a combined summary at root
    with open(os.path.join(cfg.get("results_base", "results"), "all_summaries.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    return summaries
