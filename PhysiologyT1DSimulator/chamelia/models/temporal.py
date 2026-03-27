"""Model 2: Temporal Sequence Model — lightweight transformer for BG forecasting (Block 3).

Question answered:
    Given the last N hours of feature frames, what will the next M hours of BG
    look like?

Architecture:
    Lightweight transformer encoder. Input is the hourly feature frame sequence;
    output is predicted BG summary stats for the next 4–24 hours. Therapy param
    embedding is concatenated into the sequence. Ensemble of 3–5 models with
    different random seeds provides natural uncertainty.

Why it matters:
    Powers the 'simulated preview' — a projected BG curve, not just a single TIR
    number. Catches temporal patterns the aggregate model misses. Provides a
    second opinion for ensemble agreement.
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

    class _BGTransformer(nn.Module):
        """Lightweight transformer encoder for BG sequence prediction.

        Input:  (batch, seq_len, n_features + n_actions)
        Output: (batch, horizon) predicted hourly mean BG
        """

        def __init__(
            self,
            n_features: int,
            n_actions: int = 3,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            horizon: int = 24,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.input_proj = nn.Linear(n_features + n_actions, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, horizon),
            )
            self.horizon = horizon

        def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            """Forward pass.

            Args:
                x:    (batch, seq_len, n_features + n_actions)
                mask: (batch, seq_len) bool tensor, True = padding.

            Returns:
                (batch, horizon) predicted mean BG per future hour.
            """
            h = self.input_proj(x)
            h = self.encoder(h, src_key_padding_mask=mask)
            # Use the last non-padded token's representation.
            if mask is not None:
                # Find last valid position per sample.
                lengths = (~mask).sum(dim=1).clamp(min=1)  # (batch,)
                idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h.size(-1))
                h_last = h.gather(1, idx).squeeze(1)
            else:
                h_last = h[:, -1, :]
            return self.output_head(h_last)


# ---------------------------------------------------------------------------
# PredictorCard wrapper
# ---------------------------------------------------------------------------

class TemporalSequenceModel(PredictorCard):
    """Ensemble of lightweight transformers for BG trajectory forecasting.

    Trained on (input_sequence, future_BG_sequence) pairs from the simulator.
    Ensemble of ``n_ensemble`` models with different seeds provides natural
    uncertainty: the PredictionEnvelope's lower/upper are the 10th/90th
    percentiles across ensemble members.

    Attributes (class-level):
        model_id / version / target: Standard PredictorCard metadata.
        feature_schema: Set at fit() time.
        action_schema:  [isf_multiplier, cr_multiplier, basal_multiplier].
    """

    model_id = "temporal_v1"
    version = "1.0.0"
    target = "bg_trajectory"
    feature_schema: list[str] = []
    action_schema: list[str] = ["isf_multiplier", "cr_multiplier", "basal_multiplier"]

    # Training hyperparameters.
    SEQ_LEN = 24        # Input sequence length (hours).
    HORIZON = 24        # Forecast horizon (hours).
    D_MODEL = 64
    N_HEADS = 4
    N_LAYERS = 2
    DROPOUT = 0.1
    N_ENSEMBLE = 3
    BATCH_SIZE = 64
    LR = 1e-3
    EPOCHS = 50

    def __init__(self) -> None:
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for TemporalSequenceModel. "
                "Install with: pip install torch"
            )
        self._models: list[_BGTransformer] = []
        self._n_features: int = 0
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None
        self._target_mean: float = 120.0
        self._target_std: float = 40.0
        self._fitted = False

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _make_sequences(
        X: np.ndarray,
        y_bg: np.ndarray,
        actions: np.ndarray,
        seq_len: int,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slide window over time-ordered data to produce (input, target) pairs.

        Args:
            X:       (T, n_features) hourly features.
            y_bg:    (T,) hourly mean BG.
            actions: (T, 3) therapy action per hour.
            seq_len: Input window length.
            horizon: Target window length.

        Returns:
            inputs:  (n_windows, seq_len, n_features + 3)
            targets: (n_windows, horizon)
        """
        T = X.shape[0]
        n_windows = T - seq_len - horizon + 1
        if n_windows <= 0:
            return np.empty((0, seq_len, X.shape[1] + 3)), np.empty((0, horizon))

        inputs = np.zeros((n_windows, seq_len, X.shape[1] + 3), dtype=np.float32)
        targets = np.zeros((n_windows, horizon), dtype=np.float32)

        for i in range(n_windows):
            feat_window = X[i : i + seq_len]
            act_window = actions[i : i + seq_len]
            inputs[i] = np.concatenate([feat_window, act_window], axis=1)
            targets[i] = y_bg[i + seq_len : i + seq_len + horizon]

        return inputs, targets

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_bg_train: np.ndarray,
        actions_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_bg_val: np.ndarray | None = None,
        actions_val: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        epochs: int | None = None,
        verbose: bool = True,
    ) -> "TemporalSequenceModel":
        """Train the ensemble of transformer models.

        Args:
            X_train:       (T_train, n_features) time-ordered hourly features.
            y_bg_train:    (T_train,) hourly mean BG.
            actions_train: (T_train, 3) therapy actions per hour.
            X_val / y_bg_val / actions_val: Optional validation data.
            feature_names: Column names for feature_schema.
            epochs:        Override default epoch count.
            verbose:       Print training progress.

        Returns:
            self
        """
        n_epochs = epochs or self.EPOCHS
        if feature_names is not None:
            self.feature_schema = list(feature_names)

        # Standardise features.
        self._feature_means = np.nanmean(X_train, axis=0)
        self._feature_stds = np.nanstd(X_train, axis=0)
        self._feature_stds[self._feature_stds < 1e-8] = 1.0
        self._n_features = X_train.shape[1]

        X_std = (np.nan_to_num(X_train, nan=0.0) - self._feature_means) / self._feature_stds
        self._target_mean = float(np.nanmean(y_bg_train))
        self._target_std = float(np.nanstd(y_bg_train)) + 1e-6
        y_std = (np.nan_to_num(y_bg_train, nan=self._target_mean) - self._target_mean) / self._target_std

        inputs, targets = self._make_sequences(
            X_std, y_std, np.nan_to_num(actions_train, nan=1.0),
            self.SEQ_LEN, self.HORIZON,
        )
        if inputs.shape[0] < 100:
            raise RuntimeError(
                f"Insufficient training windows: {inputs.shape[0]}. "
                "Need at least 100."
            )

        device = torch.device("cpu")
        dataset = TensorDataset(
            torch.from_numpy(inputs).float(),
            torch.from_numpy(targets).float(),
        )
        loader = DataLoader(dataset, batch_size=self.BATCH_SIZE, shuffle=True)

        # Validation set.
        val_loader = None
        if X_val is not None and y_bg_val is not None and actions_val is not None:
            X_v = (np.nan_to_num(X_val, nan=0.0) - self._feature_means) / self._feature_stds
            y_v = (np.nan_to_num(y_bg_val, nan=self._target_mean) - self._target_mean) / self._target_std
            v_in, v_tgt = self._make_sequences(
                X_v, y_v, np.nan_to_num(actions_val, nan=1.0),
                self.SEQ_LEN, self.HORIZON,
            )
            if v_in.shape[0] > 0:
                val_ds = TensorDataset(
                    torch.from_numpy(v_in).float(),
                    torch.from_numpy(v_tgt).float(),
                )
                val_loader = DataLoader(val_ds, batch_size=self.BATCH_SIZE)

        # Train ensemble.
        self._models = []
        for ens_idx in range(self.N_ENSEMBLE):
            torch.manual_seed(42 + ens_idx)
            model = _BGTransformer(
                n_features=self._n_features,
                n_actions=3,
                d_model=self.D_MODEL,
                n_heads=self.N_HEADS,
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
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * xb.size(0)
                epoch_loss /= len(dataset)

                # Validation early stopping.
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
                        if patience_counter >= 10:
                            if verbose:
                                print(f"  [temporal ens={ens_idx}] Early stop at epoch {epoch+1}")
                            break

                if verbose and (epoch + 1) % 10 == 0:
                    msg = f"  [temporal ens={ens_idx}] epoch {epoch+1}/{n_epochs} train_loss={epoch_loss:.4f}"
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
        """Predict future BG trajectory from a feature sequence.

        Args:
            features: (seq_len, n_features) or (batch, seq_len, n_features).
            action:   (3,) or (seq_len, 3) or (batch, seq_len, 3).
                      If 1-D, tiled across the sequence.

        Returns:
            PredictionEnvelope with:
                point = (horizon,) or (batch, horizon) median ensemble prediction
                lower = 10th percentile
                upper = 90th percentile
                confidence = 1 - normalised ensemble spread
        """
        if not self._fitted:
            raise RuntimeError("TemporalSequenceModel has not been fitted.")

        X = np.asarray(features, dtype=np.float32)
        if X.ndim == 2:
            X = X[np.newaxis, ...]  # (1, seq_len, n_feat)
        batch_size = X.shape[0]

        # Standardise.
        X = (np.nan_to_num(X, nan=0.0) - self._feature_means) / self._feature_stds

        # Action handling.
        if action is None:
            A = np.ones((batch_size, X.shape[1], 3), dtype=np.float32)
        else:
            A = np.asarray(action, dtype=np.float32)
            if A.ndim == 1:
                A = np.tile(A, (batch_size, X.shape[1], 1))
            elif A.ndim == 2:
                if A.shape[0] == X.shape[1]:  # (seq_len, 3)
                    A = np.tile(A[np.newaxis, ...], (batch_size, 1, 1))
                else:  # (batch, 3)
                    A = np.tile(A[:, np.newaxis, :], (1, X.shape[1], 1))

        inp = np.concatenate([X, A], axis=-1)
        inp_t = torch.from_numpy(inp).float()

        # Ensemble predictions.
        all_preds = []
        with torch.no_grad():
            for model in self._models:
                pred = model(inp_t).numpy()  # (batch, horizon)
                # Denormalise.
                pred = pred * self._target_std + self._target_mean
                all_preds.append(pred)

        preds = np.stack(all_preds, axis=0)  # (n_ensemble, batch, horizon)

        point = np.median(preds, axis=0)     # (batch, horizon)
        lower = np.percentile(preds, 10, axis=0)
        upper = np.percentile(preds, 90, axis=0)

        # Confidence: 1 - normalised ensemble spread.
        spread = np.mean(upper - lower)
        confidence = float(np.clip(1.0 - spread / (self._target_std * 2), 0.0, 1.0))

        if batch_size == 1:
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
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "feature_schema": self.feature_schema,
            "n_features": self._n_features,
            "feature_means": self._feature_means,
            "feature_stds": self._feature_stds,
            "target_mean": self._target_mean,
            "target_std": self._target_std,
            "model_states": [m.state_dict() for m in self._models],
            "config": {
                "d_model": self.D_MODEL,
                "n_heads": self.N_HEADS,
                "n_layers": self.N_LAYERS,
                "horizon": self.HORIZON,
                "dropout": self.DROPOUT,
                "n_ensemble": self.N_ENSEMBLE,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=5)

    @classmethod
    def load(cls, path: str) -> "TemporalSequenceModel":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj.feature_schema = state["feature_schema"]
        obj._n_features = state["n_features"]
        obj._feature_means = state["feature_means"]
        obj._feature_stds = state["feature_stds"]
        obj._target_mean = state["target_mean"]
        obj._target_std = state["target_std"]
        cfg = state["config"]
        obj._models = []
        for sd in state["model_states"]:
            m = _BGTransformer(
                n_features=obj._n_features,
                n_actions=3,
                d_model=cfg["d_model"],
                n_heads=cfg["n_heads"],
                n_layers=cfg["n_layers"],
                horizon=cfg["horizon"],
                dropout=cfg["dropout"],
            )
            m.load_state_dict(sd)
            m.eval()
            obj._models.append(m)
        obj._fitted = True
        return obj
