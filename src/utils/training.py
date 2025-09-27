# training/training.py
# ======================================
# 3) MODEL, TRAINING, EVALUATION
# ======================================

from __future__ import annotations
import os, csv, json
import numpy as np
import torch
import torch.nn as nn


def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    *,
    model_name: str = "LSTM",
    results_base: str = "drive/results",
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    hyperparams: dict | None = None,
    early_stop_patience: int = 15,
    early_stop_min_delta: float = 0.0,
):
    device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model.to(device)

    # --- Setup results directory ---
    model_dir = os.path.join(results_base, model_name)
    os.makedirs(model_dir, exist_ok=True)

    ckpt_last = os.path.join(model_dir, "last.pt")
    ckpt_best = os.path.join(model_dir, "best.pt")
    logs_csv  = os.path.join(model_dir, "logs.csv")

    # --- Metadata ---
    meta_path = os.path.join(model_dir, "meta.json")
    meta = {
        "model_name": model_name,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta,
    }
    if hyperparams: 
        meta.update(hyperparams)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # --- Optimizer/loss ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # --- Auto-resume ---
    start_epoch, best_val = 1, np.inf
    if os.path.exists(ckpt_last):
        state = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = state.get("epoch", 0) + 1
        best_val = state.get("best_val", np.inf)
        print(f"🔄 Resumed from {ckpt_last} at epoch {start_epoch} (best_val={best_val:.6f})")
    else:
        print(f"🆕 Fresh training for {model_name}")

    # --- Logger ---
    write_header = not os.path.exists(logs_csv)
    log_f = open(logs_csv, "a", newline="")
    logger = csv.DictWriter(
        log_f,
        fieldnames=[
            "epoch",
            "train_MSE","train_RMSE","train_%err",
            "val_MSE","val_RMSE","val_%err",
            "best_val"
        ]
    )
    if write_header:
        logger.writeheader()

    # --- Early stopping ---
    patience, min_delta, bad_epochs = early_stop_patience, early_stop_min_delta, 0

    for epoch in range(start_epoch, epochs + 1):
        # Train
        model.train()
        tr_sse, tr_sae, tr_n = 0.0, 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = model(X)
            y_hat = torch.clamp(y_hat, min=0.0)  # predictions must be non-negative

            # If model outputs (B, 1), squeeze to (B,)
            if y_hat.ndim > 1 and y_hat.size(-1) == 1:
                y_hat = y_hat.squeeze(-1)

            loss = criterion(y_hat, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tr_sse += ((y_hat - y) ** 2).sum().item()
            tr_sae += torch.abs(y_hat - y).sum().item()
            tr_n   += y.size(0)

        train_MSE  = tr_sse / tr_n
        train_RMSE = float(np.sqrt(train_MSE))
        train_perc = train_RMSE * 100.0   # % error since target ∈ [0,1]

        # Val
        model.eval()
        va_sse, va_sae, va_n = 0.0, 0.0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                y_hat = torch.clamp(y_hat, min=0.0)

                if y_hat.ndim > 1 and y_hat.size(-1) == 1:
                    y_hat = y_hat.squeeze(-1)

                va_sse += ((y_hat - y) ** 2).sum().item()
                va_sae += torch.abs(y_hat - y).sum().item()
                va_n   += y.size(0)

        val_MSE  = va_sse / va_n
        val_RMSE = float(np.sqrt(val_MSE))
        val_perc = val_RMSE * 100.0

        # Early stopping & checkpointing
        improved = val_MSE < (best_val - min_delta)
        if improved:
            best_val, bad_epochs = val_MSE, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                },
                ckpt_best,
            )
        else:
            bad_epochs += 1

        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
            },
            ckpt_last,
        )

        # Logging
        row = {
            "epoch": epoch,
            "train_MSE": train_MSE, "train_RMSE": train_RMSE, "train_%err": train_perc,
            "val_MSE": val_MSE, "val_RMSE": val_RMSE, "val_%err": val_perc,
            "best_val": best_val,
        }
        logger.writerow(row)
        log_f.flush()

        print(
            f"Epoch {epoch:03d} | "
            f"Train RMSE={train_RMSE:.4f} ({train_perc:.2f}%) | "
            f"Val RMSE={val_RMSE:.4f} ({val_perc:.2f}%) | "
            f"Best Val RMSE={np.sqrt(best_val):.4f}"
        )

        if bad_epochs >= patience:
            print("⏹️ Early stopping triggered.")
            break

    log_f.close()


def evaluate_model(
    model: torch.nn.Module,
    data_loader,
    device: str = "cuda",
    denorm=None,
):
    """
    Evaluate model and return (MAE, RMSE, preds, trues).

    Args:
        denorm: optional function to convert predictions/targets back to kW (or kWh)
                e.g., denorm(y_norm) = y_norm * capacity
                If None, metrics are in the model's target units (often normalized [0,1]).
    """
    device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model.to(device)
    model.eval()

    mae_sum, mse_sum, n = 0.0, 0.0, 0
    preds, trues = [], []

    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)

            y_hat = model(X)
            y_hat = torch.clamp(y_hat, min=0.0)

            if y_hat.ndim > 1 and y_hat.size(-1) == 1:
                y_hat = y_hat.squeeze(-1)

            if denorm is not None:
                y_np     = denorm(y.detach().cpu().numpy())
                yhat_np  = denorm(y_hat.detach().cpu().numpy())
            else:
                y_np     = y.detach().cpu().numpy()
                yhat_np  = y_hat.detach().cpu().numpy()

            preds.append(yhat_np)
            trues.append(y_np)

            mae_sum += np.abs(yhat_np - y_np).sum()
            mse_sum += ((yhat_np - y_np) ** 2).sum()
            n += len(y_np)

    preds_np = np.concatenate(preds) if preds else np.array([])
    trues_np = np.concatenate(trues) if trues else np.array([])
    mae = float(mae_sum / n) if n > 0 else float("nan")
    rmse = float(np.sqrt(mse_sum / n)) if n > 0 else float("nan")
    return mae, rmse, preds_np, trues_np
