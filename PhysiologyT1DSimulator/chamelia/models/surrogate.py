"""Model 3: BG Dynamics Surrogate — fast feedforward for counterfactual evaluation (Block 3).

Question answered:
    Given context and therapy params, what does the actual BG curve look like
    over the next 24 hours?

Architecture:
    Feedforward network that predicts parameters of a BG trajectory from daily
    context features + therapy params. Deliberately small and fast — designed
    to be called thousands of times inside an optimisation loop.

Why it matters:
    One leg of the robustness triangulation. Cheaper than the full sim, can be
    personalised as real data arrives. Powers counterfactual evaluation in the
    shadow module.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from chamelia.models.base import PredictionEnvelope, PredictorCard


# ---------------------------------------------------------------------------
# PyTorch model definition
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    class _SurrogateNet(nn.Module):
        """Feedforward network: (context + action) → 24h BG curve.

        Output is 24 predicted hourly BG values. Trained with MSE loss.
        """

        def __init__(
            self,
            n_input: int,
            hidden: int = 128,
            n_layers: int = 3,
            horizon: int = 24,
            dropout: float = 0.1,
        ):
            super().__init__()
            layers: list[nn.Module] = [nn.Linear(n_input, hidden), nn.ReLU(), nn.Dropout(dropout)]
            for _ in range(n_layers - 1):
                layers.extend([nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)])
            layers.append(nn.Linear(hidden, horizon))
            self.net = nn.Sequential(*layers)
            self.horizon = horizon

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


# ---------------------------------------------------------------------------
# PredictorCard wrapper
# ---------------------------------------------------------------------------

class BGDynamicsSurrogate(PredictorCard):
    """Fast feedforward surrogate for 24h BG curve prediction.

    Trained on daily context features + therapy settings → 24-element hourly BG
    curve. Designed for rapid evaluation inside optimisation loops (thousands of
    forward passes per cycle) and counterfactual evaluation in the shadow module.

    Ensemble of ``n_ensemble`` models provides uncertainty.
    """

    model_id = "surrogate_v1"
    version = "1.0.0"
    target = "bg_curve"
    feature_schema: list[str] = []
    action_schema: list[str] = ["isf_multiplier", "cr_multiplier", "basal_multiplier"]

    HORIZON = 24
    HIDDEN = 128
    N_LAYERS = 3
    DROPOUT = 0.1
    N_ENSEMBLE = 3
    BATCH_SIZE = 128
    LR = 1e-3
    EPOCHS = 100

    def __init__(self) -> None:
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for BGDynamicsSurrogate. "
                "Install with: pip install torch"
            )
        self._models: list[_SurrogateNet] = []
        self._n_input: int = 0
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None
        self._target_mean: np.ndarray | None = None
        self._target_std: np.ndarray | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        actions_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        actions_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        epochs: int | None = None,
        verbose: bool = True,
    ) -> "BGDynamicsSurrogate":
        """Train the ensemble of surrogate networks.

        Args:
            X_train:       (n_days, n_features) daily context features.
            y_train:       (n_days, 24) hourly BG curves.
            actions_train: (n_days, 3) therapy params [isf, cr, basal multipliers].
            X_val / y_val / actions_val: Optional validation data.
            feature_names: Column names for feature_schema.
            epochs:        Override default.
            verbose:       Print progress.

        Returns:
            self
        """
        n_epochs = epochs or self.EPOCHS
        if feature_names is not None:
            self.feature_schema = list(feature_names)

        X = np.nan_to_num(np.asarray(X_train, dtype=np.float32), nan=0.0)
        A = np.nan_to_num(np.asarray(actions_train, dtype=np.float32), nan=1.0)
        Y = np.nan_to_num(np.asarray(y_train, dtype=np.float32), nan=120.0)

        # Standardise inputs.
        XA = np.concatenate([X, A], axis=1)
        self._feature_means = np.mean(XA, axis=0)
        self._feature_stds = np.std(XA, axis=0)
        self._feature_stds[self._feature_stds < 1e-8] = 1.0
        XA_std = (XA - self._feature_means) / self._feature_stds
        self._n_input = XA_std.shape[1]

        # Standardise targets.
        self._target_mean = np.mean(Y, axis=0)
        self._target_std = np.std(Y, axis=0)
        self._target_std[self._target_std < 1e-8] = 1.0
        Y_std = (Y - self._target_mean) / self._target_std

        device = torch.device("cpu")
        dataset = TensorDataset(
            torch.from_numpy(XA_std).float(),
            torch.from_numpy(Y_std).float(),
        )
        loader = DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=True)

        # Validation.
        val_loader = None
        if X_val is not None and y_val is not None and actions_val is not None:
            Xv = np.nan_to_num(np.asarray(X_val, dtype=np.float32), nan=0.0)
            Av = np.nan_to_num(np.asarray(actions_val, dtype=np.float32), nan=1.0)
            XAv = np.concatenate([Xv, Av], axis=1)
            XAv_std = (XAv - self._feature_means) / self._feature_stds
            Yv = np.nan_to_num(np.asarray(y_val, dtype=np.float32), nan=120.0)
            Yv_std = (Yv - self._target_mean) / self._target_std
            val_ds = TensorDataset(
                torch.from_numpy(XAv_std).float(),
                torch.from_numpy(Yv_std).float(),
            )
            val_loader = DataLoader(val_ds, batch_size=self.BATCH_SIZE)

        self._models = []
        for ens_idx in range(self.N_ENSEMBLE):
            torch.manual_seed(42 + ens_idx)
            model = _SurrogateNet(
                n_input=self._n_input,
                hidden=self.HIDDEN,
                n_layers=self.N_LAYERS,
                horizon=self.HORIZON,
                dropout=self.DROPOUT,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.LR)
            criterion = nn.MSELoss()

            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(n_epochs):
                model.train()
                epoch_loss = 0.0
                for xb, yb in loader:
                    optimizer.zero_grad()
                    pred = model(xb.to(device))
                    loss = criterion(pred, yb.to(device))
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * xb.size(0)
                epoch_loss /= len(dataset)

                if val_loader is not None:
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            pred = model(xb.to(device))
                            val_loss += criterion(pred, yb.to(device)).item() * xb.size(0)
                    val_loss /= len(val_loader.dataset)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= 15:
                            if verbose:
                                print(f"  [surrogate ens={ens_idx}] Early stop at epoch {epoch+1}")
                            break

                if verbose and (epoch + 1) % 20 == 0:
                    msg = f"  [surrogate ens={ens_idx}] epoch {epoch+1}/{n_epochs} loss={epoch_loss:.4f}"
                    if val_loader is not None:
                        msg += f" val_loss={val_loss:.4f}"
                    print(msg)

            model.eval()
            self._models.append(model)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        features: np.ndarray,
        action: np.ndarray | None = None,
    ) -> PredictionEnvelope:
        """Predict 24h BG curve from context features and therapy params.

        Args:
            features: (n_features,) or (n_samples, n_features) daily context.
            action:   (3,) or (n_samples, 3). Defaults to [1, 1, 1].

        Returns:
            PredictionEnvelope with:
                point = (24,) or (n_samples, 24) predicted hourly BG
                lower = 10th percentile
                upper = 90th percentile
        """
        if not self._fitted:
            raise RuntimeError("BGDynamicsSurrogate has not been fitted.")

        X = np.asarray(features, dtype=np.float32)
        scalar_input = X.ndim == 1
        if scalar_input:
            X = X.reshape(1, -1)
        n_samples = X.shape[0]

        if action is None:
            A = np.ones((n_samples, 3), dtype=np.float32)
        else:
            A = np.asarray(action, dtype=np.float32)
            if A.ndim == 1:
                A = np.tile(A, (n_samples, 1))

        XA = np.concatenate([np.nan_to_num(X, nan=0.0), A], axis=1)
        XA_std = (XA - self._feature_means) / self._feature_stds
        inp = torch.from_numpy(XA_std).float()

        all_preds = []
        with torch.no_grad():
            for model in self._models:
                pred = model(inp).numpy()
                pred = pred * self._target_std + self._target_mean
                all_preds.append(pred)

        preds = np.stack(all_preds, axis=0)  # (n_ensemble, n_samples, 24)
        point = np.median(preds, axis=0)
        lower = np.percentile(preds, 10, axis=0)
        upper = np.percentile(preds, 90, axis=0)

        spread = np.mean(upper - lower)
        ref = float(np.mean(self._target_std))
        confidence = float(np.clip(1.0 - spread / (ref * 2), 0.0, 1.0))

        if scalar_input:
            point = point.squeeze(0)
            lower = lower.squeeze(0)
            upper = upper.squeeze(0)

        return PredictionEnvelope(
            point=point,
            lower=lower,
            upper=upper,
            confidence=confidence,
            metadata={"ensemble_size": len(self._models), "horizon": self.HORIZON},
        )

    # ------------------------------------------------------------------
    # Convenience: derive summary stats from predicted curve
    # ------------------------------------------------------------------

    def predict_summary(
        self,
        features: np.ndarray,
        action: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Predict summary glycaemic metrics from the BG curve.

        Returns dict with keys: mean_bg, percent_low, percent_high, tir, bg_var.
        """
        env = self.predict(features, action)
        curve = np.asarray(env.point)
        if curve.ndim == 1:
            curve = curve.reshape(1, -1)
        mean_bg = float(np.mean(curve))
        pct_low = float(np.mean(curve < 70))
        pct_high = float(np.mean(curve > 180))
        tir = max(0.0, 1.0 - pct_low - pct_high)
        bg_var = float(np.var(curve))
        return {
            "mean_bg": mean_bg,
            "percent_low": pct_low,
            "percent_high": pct_high,
            "tir": tir,
            "bg_var": bg_var,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "feature_schema": self.feature_schema,
            "n_input": self._n_input,
            "feature_means": self._feature_means,
            "feature_stds": self._feature_stds,
            "target_mean": self._target_mean,
            "target_std": self._target_std,
            "model_states": [m.state_dict() for m in self._models],
            "config": {
                "hidden": self.HIDDEN,
                "n_layers": self.N_LAYERS,
                "horizon": self.HORIZON,
                "dropout": self.DROPOUT,
                "n_ensemble": self.N_ENSEMBLE,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=5)

    @classmethod
    def load(cls, path: str) -> "BGDynamicsSurrogate":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj.feature_schema = state["feature_schema"]
        obj._n_input = state["n_input"]
        obj._feature_means = state["feature_means"]
        obj._feature_stds = state["feature_stds"]
        obj._target_mean = state["target_mean"]
        obj._target_std = state["target_std"]
        cfg = state["config"]
        obj._models = []
        for sd in state["model_states"]:
            m = _SurrogateNet(
                n_input=obj._n_input,
                hidden=cfg["hidden"],
                n_layers=cfg["n_layers"],
                horizon=cfg["horizon"],
                dropout=cfg["dropout"],
            )
            m.load_state_dict(sd)
            m.eval()
            obj._models.append(m)
        obj._fitted = True
        return obj
