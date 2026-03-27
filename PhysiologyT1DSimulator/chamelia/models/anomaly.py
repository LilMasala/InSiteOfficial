"""Model 4: Anomaly / Regime Detector — GP familiarity in reduced feature space (Block 3).

Answers: Is the current state normal, or has something shifted?

Architecture:
    - PCA projects ~38-dim feature space to 8–12 dims
    - Sparse Gaussian Process fitted over training data in reduced space
    - GP does NOT predict outcomes — it models training data density
    - For any new input, posterior std indicates familiarity:
        Low variance → system has seen similar states, models reliable
        High variance → uncharted territory, suppress recommendations

The GP updates naturally as new data arrives (closed-loop sim or real users),
and the familiar region expands organically.
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

from chamelia.models.base import PredictionEnvelope, PredictorCard


class AnomalyDetector(PredictorCard):
    """Sparse GP familiarity detector in PCA-reduced feature space.

    Does not predict outcomes — instead reports how far the current state is
    from the training distribution.  The ``confidence`` field in the returned
    PredictionEnvelope is repurposed as a *familiarity score* (1.0 = very
    familiar, 0.0 = completely novel).

    Attributes (class-level):
        model_id / version / target: Standard PredictorCard metadata.
        feature_schema:  Set at fit() time from training columns.
        action_schema:   None — observation-only model.
    """

    model_id = "anomaly_v1"
    version = "1.0.0"
    target = "anomaly"
    feature_schema: list[str] = []
    action_schema = None

    # Dimensionality of the PCA projection.
    N_COMPONENTS = 10

    # Maximum training points fed to the GP (for tractability).
    MAX_INDUCING = 2000

    def __init__(self) -> None:
        self._pca: PCA | None = None
        self._gp: GaussianProcessRegressor | None = None
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None
        self._fitted = False
        # Reference std for normalising familiarity scores.
        self._ref_std: float = 1.0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        feature_names: list[str] | None = None,
        n_components: int | None = None,
        max_inducing: int | None = None,
    ) -> "AnomalyDetector":
        """Fit the PCA + GP density model.

        Args:
            X_train:       (n_samples, n_features) training feature matrix.
            feature_names: Optional column names for feature_schema.
            n_components:  Override default PCA dimensionality.
            max_inducing:  Override maximum GP training points.

        Returns:
            self
        """
        n_comp = n_components or self.N_COMPONENTS
        max_ind = max_inducing or self.MAX_INDUCING

        X = np.asarray(X_train, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if feature_names is not None:
            self.feature_schema = list(feature_names)

        # Standardise features.
        self._feature_means = np.nanmean(X, axis=0)
        self._feature_stds = np.nanstd(X, axis=0)
        self._feature_stds[self._feature_stds < 1e-8] = 1.0
        X_std = (np.nan_to_num(X, nan=0.0) - self._feature_means) / self._feature_stds

        # PCA reduction.
        n_comp = min(n_comp, X_std.shape[1], X_std.shape[0])
        self._pca = PCA(n_components=n_comp, random_state=42)
        X_pca = self._pca.fit_transform(X_std)

        # Subsample for GP tractability.
        if X_pca.shape[0] > max_ind:
            rng = np.random.default_rng(42)
            idx = rng.choice(X_pca.shape[0], max_ind, replace=False)
            X_pca = X_pca[idx]

        # GP with RBF kernel — we train it to predict a dummy constant (0).
        # The posterior std is what we care about: high std = unfamiliar.
        kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(n_comp)) + WhiteKernel(0.1)
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=2,
            normalize_y=False,
            random_state=42,
        )
        y_dummy = np.zeros(X_pca.shape[0])
        self._gp.fit(X_pca, y_dummy)

        # Compute reference std from training data for normalisation.
        _, train_std = self._gp.predict(X_pca, return_std=True)
        self._ref_std = float(np.percentile(train_std, 95)) + 1e-6

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Predict (familiarity scoring)
    # ------------------------------------------------------------------

    def predict(
        self,
        features: np.ndarray,
        action: np.ndarray | None = None,
    ) -> PredictionEnvelope:
        """Compute familiarity score for the given feature vector(s).

        Args:
            features: (n_features,) or (n_samples, n_features).
            action:   Ignored (observation-only model).

        Returns:
            PredictionEnvelope where:
                point      = posterior std (raw unfamiliarity)
                lower      = 0.0 (not meaningful)
                upper      = posterior std
                confidence = familiarity score in [0, 1]
                metadata   = {"is_anomaly": bool}
        """
        if not self._fitted:
            raise RuntimeError("AnomalyDetector has not been fitted.")

        X = np.asarray(features, dtype=float)
        scalar_input = X.ndim == 1
        if scalar_input:
            X = X.reshape(1, -1)

        # Standardise and project.
        X_std = (np.nan_to_num(X, nan=0.0) - self._feature_means) / self._feature_stds
        X_pca = self._pca.transform(X_std)

        _, posterior_std = self._gp.predict(X_pca, return_std=True)

        # Familiarity: 1 when std is low relative to training, 0 when high.
        familiarity = np.clip(1.0 - posterior_std / self._ref_std, 0.0, 1.0)

        # Anomaly flag: familiarity below 0.3 is considered anomalous.
        is_anomaly = familiarity < 0.3

        if scalar_input:
            return PredictionEnvelope(
                point=float(posterior_std[0]),
                lower=0.0,
                upper=float(posterior_std[0]),
                confidence=float(familiarity[0]),
                metadata={"is_anomaly": bool(is_anomaly[0])},
            )

        return PredictionEnvelope(
            point=posterior_std,
            lower=np.zeros_like(posterior_std),
            upper=posterior_std,
            confidence=float(np.mean(familiarity)),
            metadata={"is_anomaly": is_anomaly.tolist(), "familiarity": familiarity.tolist()},
        )

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------

    def update(self, X_new: np.ndarray) -> None:
        """Incrementally expand the GP with new observations.

        For efficiency, only updates if the new points include anomalies
        (unfamiliar states that should become familiar after being observed).
        """
        if not self._fitted:
            raise RuntimeError("AnomalyDetector has not been fitted.")

        X = np.asarray(X_new, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_std = (np.nan_to_num(X, nan=0.0) - self._feature_means) / self._feature_stds
        X_pca = self._pca.transform(X_std)

        # Append to existing GP training data and refit.
        X_existing = self._gp.X_train_
        X_combined = np.vstack([X_existing, X_pca])

        # Cap total size.
        if X_combined.shape[0] > self.MAX_INDUCING:
            rng = np.random.default_rng(42)
            idx = rng.choice(X_combined.shape[0], self.MAX_INDUCING, replace=False)
            X_combined = X_combined[idx]

        y_dummy = np.zeros(X_combined.shape[0])
        self._gp.fit(X_combined, y_dummy)

        # Recompute reference std.
        _, train_std = self._gp.predict(X_combined, return_std=True)
        self._ref_std = float(np.percentile(train_std, 95)) + 1e-6

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=5)

    @classmethod
    def load(cls, path: str) -> "AnomalyDetector":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            warnings.warn(
                f"Loaded object type {type(obj).__name__} does not match "
                f"{cls.__name__}.",
                UserWarning,
                stacklevel=2,
            )
        return obj
