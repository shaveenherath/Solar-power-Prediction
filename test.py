# test.py
from __future__ import annotations
import argparse, os, json
from typing import Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.utils.data_loading import create_dataloaders_grouped
from src.dataset import SolarDatasetGrouped
from src.models import LSTMForecast, TimeSeriesTransformer, CNNTransformerHybrid, LLaMATimeSeries

MODEL_REGISTRY = {
    "lstm": LSTMForecast,
    "transformer": TimeSeriesTransformer,
    "cnn_transformer": CNNTransformerHybrid,
    "llama_ts": LLaMATimeSeries,
}

def _load_config(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".yaml", ".yml"]:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise SystemExit(f"Unsupported config extension '{ext}'. Use .yaml/.yml/.json")

def _device(pref: str = "cuda") -> str:
    return pref if (pref == "cuda" and torch.cuda.is_available()) else "cpu"

def _init_model(key: str, input_size: int, cfg: Dict[str, Any]) -> torch.nn.Module:
    key = key.lower()
    if key == "lstm":
        return MODEL_REGISTRY[key](
            input_size=input_size,
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.2),
        )
    if key == "transformer":
        return MODEL_REGISTRY[key](
            input_size=input_size,
            num_heads=cfg.get("num_heads", 8),
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 6),
            dropout=cfg.get("dropout", 0.1),
        )
    if key == "cnn_transformer":
        return MODEL_REGISTRY[key](
            input_size=input_size,
            cnn_out_channels=cfg.get("cnn_out_channels", 64),
            kernel_size=cfg.get("kernel_size", 3),
            num_heads=cfg.get("num_heads", 8),
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 6),
            dropout=cfg.get("dropout", 0.1),
        )
    if key == "llama_ts":
        return MODEL_REGISTRY[key](
            n_features=input_size,
            d_model=cfg.get("d_model", 128),
            n_heads=cfg.get("n_heads", 8),
            n_layers=cfg.get("n_layers", 4),
            dropout=cfg.get("dropout", 0.1),
            max_seq_len=cfg.get("max_seq_len", 48),
        )
    raise ValueError(f"Unknown model key: {key}")

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

@torch.no_grad()
def _predict_all(model: torch.nn.Module, test_loader, device: str):
    model.to(device)
    model.eval()
    preds, trues = [], []
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        # strict clamp as requested (evaluation-time; training already handled)
        y_pred = torch.clamp(y_pred, min=0.0)
        if y_pred.ndim > 1 and y_pred.size(-1) == 1:
            y_pred = y_pred.squeeze(-1)
        preds.append(y_pred.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)

def _plot_series(path: str, y_true, y_pred, title="Actual vs Predicted"):
    _ensure_dir(os.path.dirname(path))
    plt.figure(figsize=(14,5))
    plt.plot(y_true, label="Actual", alpha=0.7)
    plt.plot(y_pred, label="Predicted", alpha=0.7)
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Normalized Solar Generation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def _plot_zoom(path: str, y_true, y_pred, start=0, length=500, title="Zoomed"):
    _ensure_dir(os.path.dirname(path))
    end = min(len(y_true), start+length)
    plt.figure(figsize=(14,5))
    plt.plot(y_true[start:end], label="Actual")
    plt.plot(y_pred[start:end], label="Predicted")
    plt.title(f"{title} [{start}:{end}]")
    plt.xlabel("Time step")
    plt.ylabel("Normalized Solar Generation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate per-model + ensemble (clamp-based metrics)")
    parser.add_argument("--config", type=str, required=True, help="Path to config (.yaml/.yml/.json)")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    device = _device(cfg.get("device", "cuda"))
    results_base = cfg.get("results_base", "results")
    plots_dir = os.path.join(results_base, "plots")
    compare_dir = os.path.join(results_base, "compare")
    _ensure_dir(plots_dir); _ensure_dir(compare_dir)

    # Build loaders (we only need test_loader, but create all for consistency)
    train_loader, val_loader, test_loader, scaler, (idx_tr, idx_va, idx_te) = create_dataloaders_grouped(
        csv_file=cfg["dataloaders"]["csv_file"],
        feature_cols=cfg["dataloaders"]["feature_cols"],
        target_col=cfg["dataloaders"].get("target_col", "y_norm"),
        lookback=cfg["dataloaders"].get("lookback", 24),
        horizon=cfg["dataloaders"].get("horizon", 1),
        batch_size=cfg["dataloaders"].get("batch_size", 128),
        train_frac=cfg["dataloaders"].get("train_frac", 0.7),
        val_frac=cfg["dataloaders"].get("val_frac", 0.2),
        num_workers=cfg["dataloaders"].get("num_workers", 0),
        pin_memory=cfg["dataloaders"].get("pin_memory", False),
        shuffle_train=False,
    )

    # For plotting with timestamps if needed:
    ds_all = SolarDatasetGrouped(
        csv_file=cfg["dataloaders"]["csv_file"],
        feature_cols=cfg["dataloaders"]["feature_cols"],
        target_col=cfg["dataloaders"].get("target_col", "y_norm"),
        lookback=cfg["dataloaders"].get("lookback", 24),
        horizon=cfg["dataloaders"].get("horizon", 1),
        scaler=scaler,
    )
    # test_timestamps = ds_all.seq_t[idx_te]  # not used in current plots (index-based)

    run_which = cfg.get("run", {}).get("model", "all")
    model_cfgs = cfg.get("models", {})
    model_keys = list(model_cfgs.keys()) if run_which == "all" else [run_which]

    input_size = len(cfg["dataloaders"]["feature_cols"])

    all_preds: Dict[str, np.ndarray] = {}
    ytrue_all = None
    results_list = []

    # --- Evaluate each model (using best.pt) ---
    for key in model_keys:
        mcfg = model_cfgs.get(key, {})
        model_name = mcfg.get("name", key)
        model = _init_model(key, input_size, mcfg)

        ckpt_dir = os.path.join(results_base, model_name)
        best_ckpt = os.path.join(ckpt_dir, "best.pt")
        if not os.path.exists(best_ckpt):
            raise FileNotFoundError(f"Missing checkpoint for {model_name}: {best_ckpt}")

        state = torch.load(best_ckpt, map_location=device) if hasattr(torch.load, "__call__") else torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["model_state"], strict=False)

        y_pred_all, y_true = _predict_all(model, test_loader, device=device)
        if ytrue_all is None:
            ytrue_all = y_true

        all_preds[model_name] = y_pred_all

        mae = float(np.mean(np.abs(y_pred_all - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred_all - y_true) ** 2)))
        # your requested %Error definition:
        perc_err = float((rmse / max(1e-12, y_true.mean())) * 100.0)

        print(f"{model_name} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | %Error: {perc_err:.2f}%")
        results_list.append({"Model": model_name, "MAE": mae, "RMSE": rmse, "%Error": perc_err})

        # Plots per model
        _plot_series(os.path.join(plots_dir, f"{model_name}_full.png"),
                     y_true, y_pred_all, title=f"{model_name}: Actual vs Predicted (Full)")
        zoom_cfg = cfg.get("plot", {})
        _plot_zoom(os.path.join(plots_dir, f"{model_name}_zoom.png"),
                   y_true, y_pred_all,
                   start=int(zoom_cfg.get("zoom_start_idx", 0)),
                   length=int(zoom_cfg.get("zoom_length", 500)),
                   title=f"{model_name}: Actual vs Predicted (Zoom)")

    # --- Ensemble (simple average over available model preds) ---
    if len(all_preds) >= 2:
        stacked = np.stack([pred for pred in all_preds.values()], axis=0)  # [M, N]
        y_ens = stacked.mean(axis=0)
        mae = float(np.mean(np.abs(y_ens - ytrue_all)))
        rmse = float(np.sqrt(np.mean((y_ens - ytrue_all) ** 2)))
        perc_err = float((rmse / max(1e-12, ytrue_all.mean())) * 100.0)
        print(f"\n✅ Ensemble | MAE: {mae:.4f} | RMSE: {rmse:.4f} | %Error: {perc_err:.2f}%")
        results_list.append({"Model": "Ensemble", "MAE": mae, "RMSE": rmse, "%Error": perc_err})

        # Ensemble plots
        _plot_series(os.path.join(plots_dir, "Ensemble_full.png"),
                     ytrue_all, y_ens, title="Ensemble: Actual vs Predicted (Full)")
        zoom_cfg = cfg.get("plot", {})
        _plot_zoom(os.path.join(plots_dir, "Ensemble_zoom.png"),
                   ytrue_all, y_ens,
                   start=int(zoom_cfg.get("zoom_start_idx", 0)),
                   length=int(zoom_cfg.get("zoom_length", 500)),
                   title="Ensemble: Actual vs Predicted (Zoom)")
    else:
        print("\nEnsemble skipped (need >= 2 models).")

    # --- Save comparison CSV (sorted by RMSE) ---
    compare_csv = os.path.join(compare_dir, "ensemble_model_comparison.csv")
    results_df = pd.DataFrame(results_list).sort_values("RMSE")
    results_df.to_csv(compare_csv, index=False)
    print(f"\n✅ Results saved: {compare_csv}")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
