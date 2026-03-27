"""AggregateOutcomePredictor — Model 1 in the Chamelia zoo (Block 3)."""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from chamelia.models.base import PredictionEnvelope, PredictorCard


# XGBoost params shared across all heads.
_BASE_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "random_state": 42,
    "verbosity": 0,
}


class AggregateOutcomePredictor(PredictorCard):
    """XGBoost multi-output predictor for daily glycaemic outcomes.

    Answers the core optimisation question:
        "Given today's feature context and a proposed therapy setting, what
         will the outcome metrics be?"

    Targets:
        percent_low   — fraction of CGM readings below 70 mg/dL
        percent_high  — fraction of CGM readings above 180 mg/dL
        tir           — time in range [70, 180] mg/dL
        mean_bg       — mean BG (mg/dL)

    Inputs:
        feature_schema columns (daily-aggregated feature_frames) +
        action_schema  = [isf_multiplier, cr_multiplier, basal_multiplier]

    Architecture:
        4 targets × 3 quantiles (q=0.1, 0.5, q=0.9) = 12 XGBRegressor models.
        The q=0.5 model serves as the point estimate.

    Uncertainty:
        confidence = 1 − mean( (upper − lower) / (upper + lower + ε) ), clipped
        to [0, 1].  A narrow interval relative to the predicted value gives
        high confidence; a stub or wildly uncertain model gives confidence → 0.
    """

    model_id = "aggregate_v1"
    version = "1.0.0"
    target = "aggregate"
    action_schema: list[str] = ["isf_multiplier", "cr_multiplier", "basal_multiplier"]

    TARGETS: list[str] = ["percent_low", "percent_high", "tir", "mean_bg"]
    QUANTILES: list[float] = [0.1, 0.5, 0.9]

    # Index into TARGETS for the point-estimate model (median).
    _MEDIAN_IDX = 1

    def __init__(self) -> None:
        self.feature_schema: list[str] = []
        # _models[target_idx][quantile_idx] = XGBRegressor
        self._models: list[list[XGBRegressor]] = []
        self._site_encoder: LabelEncoder = LabelEncoder()
        self._feature_means: pd.Series = pd.Series(dtype=float)
        self._site_col_idx: int = -1   # column index of site_loc_current_enc in feature matrix
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
        xgb_params: dict | None = None,
    ) -> "AggregateOutcomePredictor":
        """Train 12 XGBRegressor models (4 targets × 3 quantiles).

        Args:
            X_train:    Training feature matrix.  Must include
                        ``site_loc_current_enc`` (integer-encoded) as one
                        column.  All other columns must be numeric.
            y_train:    DataFrame with columns matching ``TARGETS``.
            X_val:      Optional validation features for early stopping.
            y_val:      Optional validation labels for early stopping.
            xgb_params: Override default XGBoost hyper-parameters.

        Returns:
            self (for method chaining).
        """
        if len(X_train) < 100:
            raise RuntimeError(
                f"Insufficient training rows: {len(X_train)}.  "
                "Need at least 100 rows."
            )

        params = {**_BASE_PARAMS, **(xgb_params or {})}
        # feature_schema contains ONLY the non-action columns so that
        # predict(features, action) can safely concatenate them.
        self.feature_schema = [
            c for c in X_train.columns if c not in self.action_schema
        ]

        # Locate site encoding column for predict().
        if "site_loc_current_enc" in self.feature_schema:
            self._site_col_idx = self.feature_schema.index("site_loc_current_enc")

        # Imputation baseline (computed from training set only).
        self._feature_means = X_train.select_dtypes(include="number").mean()

        X_tr = X_train.fillna(self._feature_means).values.astype(float)
        if X_val is not None:
            X_v = X_val.fillna(self._feature_means).values.astype(float)

        self._models = []
        for t_idx, target in enumerate(self.TARGETS):
            target_models: list[XGBRegressor] = []
            y_tr_col = y_train[target].values.astype(float)
            y_v_col = y_val[target].values.astype(float) if y_val is not None else None

            for q in self.QUANTILES:
                if q == 0.5:
                    obj = "reg:squarederror"
                    model = XGBRegressor(objective=obj, **params)
                else:
                    obj = "reg:quantileerror"
                    model = XGBRegressor(
                        objective=obj,
                        quantile_alpha=q,
                        **{k: v for k, v in params.items() if k != "objective"},
                    )

                fit_kwargs: dict = {}
                if X_val is not None and y_v_col is not None:
                    model.set_params(early_stopping_rounds=30)
                    fit_kwargs["eval_set"] = [(X_v, y_v_col)]
                    fit_kwargs["verbose"] = False

                model.fit(X_tr, y_tr_col, **fit_kwargs)
                target_models.append(model)

            self._models.append(target_models)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        features: np.ndarray | pd.DataFrame,
        action: np.ndarray | None = None,
    ) -> PredictionEnvelope:
        """Predict glycaemic outcomes for *features* under *action*.

        Args:
            features: Shape (n_features,) for a single sample or
                      (n_samples, n_features) for a batch.  Column order must
                      match ``feature_schema`` (excluding action columns).
                      May include ``site_loc_current_enc`` as an integer column.
            action:   Shape (3,) or (n_samples, 3): [isf_mult, cr_mult, basal_mult].
                      If None, action columns are set to 1.0.

        Returns:
            PredictionEnvelope with arrays of shape (4,) (single sample) or
            (n_samples, 4) for the four TARGETS.
        """
        if not self._fitted:
            raise RuntimeError("Model has not been fitted.  Call fit() first.")

        # Coerce to 2-D array.
        if isinstance(features, pd.DataFrame):
            X = features.fillna(self._feature_means).values.astype(float)
        else:
            X = np.asarray(features, dtype=float)
        scalar_input = X.ndim == 1
        if scalar_input:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]

        # Append action columns.
        if action is not None:
            A = np.asarray(action, dtype=float)
            if A.ndim == 1:
                A = np.tile(A, (n_samples, 1))
            A = np.clip(A, 0.70, 1.35)
        else:
            A = np.ones((n_samples, 3), dtype=float)

        # Impute NaN in feature portion.
        if self._feature_means.size > 0:
            for col_idx, col_name in enumerate(self.feature_schema):
                if col_name in self._feature_means.index and col_idx < X.shape[1]:
                    nan_mask = ~np.isfinite(X[:, col_idx])
                    if nan_mask.any():
                        X[nan_mask, col_idx] = float(self._feature_means[col_name])

        X_full = np.concatenate([X, A], axis=1)

        unseen_site_label = False
        meta: dict = {"targets": self.TARGETS}

        # Run all 12 models.
        # Results: shape (n_targets, n_quantiles, n_samples)
        preds = np.zeros((len(self.TARGETS), len(self.QUANTILES), n_samples), dtype=float)
        importances: dict[str, np.ndarray] = {}
        for t_idx, (target, models) in enumerate(zip(self.TARGETS, self._models)):
            for q_idx, model in enumerate(models):
                preds[t_idx, q_idx] = model.predict(X_full)
            # Feature importances from the median model.
            importances[target] = models[self._MEDIAN_IDX].feature_importances_

        meta["feature_importances"] = importances
        if unseen_site_label:
            meta["unseen_site_label"] = True

        # point = q=0.5 (MEDIAN_IDX=1), lower = q=0.1, upper = q=0.9
        point = preds[:, self._MEDIAN_IDX, :]   # (4, n_samples)
        lower = preds[:, 0, :]
        upper = preds[:, 2, :]

        # Transpose to (n_samples, 4) then squeeze single-sample.
        point = point.T
        lower = lower.T
        upper = upper.T

        # Confidence: 1 - mean normalised interval width.
        interval = np.abs(upper - lower)
        denom = np.abs(upper) + np.abs(lower) + 1e-6
        confidence = float(np.clip(1.0 - np.mean(interval / denom), 0.0, 1.0))

        if scalar_input:
            point = point.squeeze(0)   # (4,)
            lower = lower.squeeze(0)
            upper = upper.squeeze(0)

        return PredictionEnvelope(
            point=point,
            lower=lower,
            upper=upper,
            confidence=confidence,
            metadata=meta,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle the full model to *path*."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=5)

    @classmethod
    def load(cls, path: str) -> "AggregateOutcomePredictor":
        """Load and return a previously saved AggregateOutcomePredictor."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            warnings.warn(
                f"Loaded object type {type(obj).__name__} does not match "
                f"{cls.__name__}.  Proceeding anyway.",
                UserWarning,
                stacklevel=2,
            )
        return obj
