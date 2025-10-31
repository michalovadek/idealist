"""
Ideal Point Estimation with multiple inference backends.

This module provides a unified interface for estimating ideal points (latent positions)
from binary, ordinal, continuous, count, or bounded continuous response data. Ideal point
models estimate latent positions of persons and items on one or more dimensions across
various applications including political science, psychometrics, marketing, and social science.

The model supports three inference options:
- MAP: Fast point estimation (~seconds)
- VI: Fast variational inference (~seconds)
- MCMC: Full Bayesian inference (~minutes)

Model Specification:
------------------
For binary responses:
    P(y_ij = 1) = σ(α_j + β_j · θ_i)

Where:
    - θ_i: Ideal point for person i (n_dims dimensional)
    - α_j: Item difficulty/intercept
    - β_j: Item discrimination (n_dims dimensional)
    - σ: Logistic function

Extensions:
-----------
- Multi-dimensional ideal points
- Temporal dynamics (ideal points evolving over time)
- Person and item covariates
- Multiple response types (binary, ordinal, continuous, count)
"""

import time
from typing import Optional, Dict, Union, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal

from ..core.base import BaseIdealPointModel, IdealPointConfig, IdealPointResults, ResponseType
from ..core.device import DeviceManager
from ..utils.logging import get_logger

logger = get_logger(__name__)


class IdealPointEstimator(BaseIdealPointModel):
    """
    Unified ideal point estimation model with flexible inference.

    This class provides a single model definition with three inference options:

    1. **MAP** (Maximum A Posteriori): Fast point estimation with Laplace approximation
       - Speed: ~seconds
       - Uncertainty: Laplace approximation at mode
       - Use for: Production, large datasets, quick iterations

    2. **VI** (Variational Inference): Fast approximate Bayesian inference
       - Speed: ~seconds to minutes
       - Uncertainty: Approximate posterior via variational family
       - Use for: Good balance of speed and uncertainty quantification

    3. **MCMC** (Markov Chain Monte Carlo): Full Bayesian inference
       - Speed: ~minutes to hours
       - Uncertainty: True posterior via sampling
       - Use for: Research, complex models, gold standard inference

    Parameters
    ----------
    config : IdealPointConfig
        Model configuration specifying dimensions, priors, response type, etc.

    Examples
    --------
    >>> config = IdealPointConfig(n_dims=1, response_type=ResponseType.BINARY)
    >>> model = IdealPointEstimator(config)
    >>>
    >>> # Fast inference (seconds)
    >>> results = model.fit(person_ids, item_ids, responses, inference='vi')
    >>>
    >>> # Full Bayesian inference (minutes)
    >>> results = model.fit(person_ids, item_ids, responses, inference='mcmc')
    """

    def __init__(self, config: Optional[IdealPointConfig] = None):
        super().__init__(config)
        self.posterior_samples = None
        self.mcmc = None
        self.svi = None
        self.guide = None

    def _build_model(
        self,
        person_ids: jnp.ndarray,
        item_ids: jnp.ndarray,
        responses: Optional[jnp.ndarray] = None,
        person_covariates: Optional[jnp.ndarray] = None,
        item_covariates: Optional[jnp.ndarray] = None,
        timepoints: Optional[jnp.ndarray] = None,
    ):
        """
        NumPyro model definition for ideal point estimation.

        Single source of truth for the ideal point model, used by all inference methods.
        Defines the generative process for responses given ideal points and item parameters.
        """
        n_persons = self.config.n_persons
        n_items = self.config.n_items
        n_dims = self.config.n_dims

        # ===== IDEAL POINTS =====
        if self.config.temporal_dynamics and timepoints is not None:
            # Temporal model: θ_t = θ_{t-1} + ε_t
            if self.config.n_timepoints is None:
                raise ValueError("config.n_timepoints must be set for temporal models")

            # Initial positions
            theta_0 = numpyro.sample(
                "theta_initial",
                dist.Normal(
                    0.0,
                    self.config.prior_ideal_point_scale
                ).expand([n_persons, n_dims]).to_event(2)
            )

            # Random walk innovations
            innovations = numpyro.sample(
                "theta_innovations",
                dist.Normal(0.0, self.config.temporal_variance)
                .expand([self.config.n_timepoints - 1, n_persons, n_dims])
                .to_event(3)
            )

            # Reconstruct trajectory
            cumulative = jnp.cumsum(innovations, axis=0)
            theta_all = jnp.concatenate([
                theta_0[None, :, :],
                theta_0[None, :, :] + cumulative
            ], axis=0)

            # Index by timepoint
            ideal_points = theta_all[timepoints, person_ids, :]

        else:
            # Static model
            if self.config.hierarchical and person_covariates is not None:
                # Hierarchical: θ_i ~ N(X_i·γ, σ²)
                n_covariates = person_covariates.shape[1]
                gamma = numpyro.sample(
                    "person_covariate_effects",
                    dist.Normal(0.0, self.config.prior_covariate_scale)
                    .expand([n_covariates, n_dims])
                )
                mean_theta = jnp.dot(person_covariates, gamma)
                ideal_points_all = numpyro.sample(
                    "ideal_points",
                    dist.Normal(mean_theta, self.config.prior_ideal_point_scale).to_event(2)
                )
            else:
                # Standard: θ_i ~ N(0, σ²)
                ideal_points_all = numpyro.sample(
                    "ideal_points",
                    dist.Normal(
                        self.config.prior_ideal_point_mean,
                        self.config.prior_ideal_point_scale
                    ).expand([n_persons, n_dims]).to_event(2)
                )

            ideal_points = ideal_points_all[person_ids, :]

        # ===== ITEM PARAMETERS =====
        # Difficulty: α_j ~ N(0, σ_α²)
        if self.config.hierarchical and item_covariates is not None:
            n_covariates = item_covariates.shape[1]
            delta = numpyro.sample(
                "item_difficulty_covariate_effects",
                dist.Normal(0.0, self.config.prior_covariate_scale).expand([n_covariates])
            )
            mean_alpha = jnp.dot(item_covariates, delta)
            difficulty = numpyro.sample(
                "difficulty",
                dist.Normal(mean_alpha, self.config.prior_difficulty_scale).to_event(1)
            )
        else:
            difficulty = numpyro.sample(
                "difficulty",
                dist.Normal(
                    self.config.prior_difficulty_mean,
                    self.config.prior_difficulty_scale
                ).expand([n_items]).to_event(1)
            )

        # Discrimination: β_j ~ N(0, σ_β²)
        if self.config.hierarchical and item_covariates is not None:
            n_covariates = item_covariates.shape[1]
            zeta = numpyro.sample(
                "item_discrimination_covariate_effects",
                dist.Normal(0.0, self.config.prior_covariate_scale)
                .expand([n_covariates, n_dims])
            )
            mean_beta = jnp.dot(item_covariates, zeta)
            discrimination = numpyro.sample(
                "discrimination",
                dist.Normal(mean_beta, self.config.prior_discrimination_scale).to_event(2)
            )
        else:
            discrimination = numpyro.sample(
                "discrimination",
                dist.Normal(
                    self.config.prior_discrimination_mean,
                    self.config.prior_discrimination_scale
                ).expand([n_items, n_dims]).to_event(2)
            )

        # Index by item
        difficulty_i = difficulty[item_ids]
        discrimination_i = discrimination[item_ids, :]

        # ===== LINEAR PREDICTOR =====
        # η = α_j + β_j·θ_i
        interaction = jnp.sum(discrimination_i * ideal_points, axis=-1)
        linear_pred = difficulty_i + interaction

        # ===== LIKELIHOOD =====
        if self.config.response_type == ResponseType.BINARY:
            with numpyro.plate("data", len(linear_pred)):
                numpyro.sample("obs", dist.Bernoulli(logits=linear_pred), obs=responses)

        elif self.config.response_type == ResponseType.ORDINAL:
            thresholds = numpyro.sample(
                "thresholds",
                dist.TransformedDistribution(
                    dist.Normal(0.0, self.config.prior_threshold_scale)
                    .expand([self.config.n_categories - 1]),
                    dist.transforms.OrderedTransform()
                )
            )
            cumulative_probs = jax.nn.sigmoid(thresholds[None, :] - linear_pred[:, None])
            padded = jnp.concatenate([
                jnp.zeros((cumulative_probs.shape[0], 1)),
                cumulative_probs,
                jnp.ones((cumulative_probs.shape[0], 1))
            ], axis=1)
            probs = padded[:, 1:] - padded[:, :-1]
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=responses)

        elif self.config.response_type == ResponseType.CONTINUOUS:
            sigma = numpyro.sample("sigma", dist.HalfNormal(self.config.prior_residual_scale))
            numpyro.sample("obs", dist.Normal(linear_pred, sigma), obs=responses)

        elif self.config.response_type == ResponseType.COUNT:
            rate = jnp.exp(linear_pred)
            numpyro.sample("obs", dist.Poisson(rate), obs=responses)

        elif self.config.response_type == ResponseType.BOUNDED_CONTINUOUS:
            precision = numpyro.sample(
                "precision",
                dist.Gamma(
                    self.config.prior_precision_shape,
                    self.config.prior_precision_rate
                )
            )
            epsilon = 1e-6
            y_scaled = (responses - self.config.response_lower_bound) / \
                       (self.config.response_upper_bound - self.config.response_lower_bound)
            y_scaled = jnp.clip(y_scaled, epsilon, 1 - epsilon)
            mu = jax.nn.sigmoid(linear_pred)
            mu = jnp.clip(mu, epsilon, 1 - epsilon)
            alpha = mu * precision
            beta = (1 - mu) * precision
            with numpyro.plate("data", len(responses)):
                numpyro.sample("obs", dist.Beta(alpha, beta), obs=y_scaled)

    def fit(
        self,
        # Data (can pass IdealPointData or raw arrays)
        data: Union['IdealPointData', np.ndarray] = None,
        person_ids: Optional[np.ndarray] = None,
        item_ids: Optional[np.ndarray] = None,
        responses: Optional[np.ndarray] = None,
        person_covariates: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        item_covariates: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        timepoints: Optional[np.ndarray] = None,
        # Inference method
        inference: Literal['vi', 'mcmc', 'map'] = 'vi',
        # Inference-specific parameters
        num_samples: int = 1000,
        num_warmup: int = 500,
        num_chains: int = 1,
        vi_steps: int = 10000,
        vi_optimizer: str = 'adam',
        vi_lr: float = 0.01,
        map_steps: int = 5000,
        map_optimizer: str = 'adam',
        map_lr: float = 0.01,
        # Runtime/hardware
        device: str = 'auto',
        progress_bar: bool = True,
        random_seed: Optional[int] = None,
        # Advanced
        chain_method: str = 'parallel',
        guide_type: str = 'normal',
        **kwargs
    ) -> IdealPointResults:
        """
        Fit the ideal point model using specified inference method.

        Parameters
        ----------
        data : IdealPointData, optional
            Data object from load_data(). If provided, person_ids/item_ids/responses are ignored.
        person_ids : np.ndarray, shape (n_obs,), optional
            Person indices (0 to n_persons-1). Required if data is not provided.
        item_ids : np.ndarray, shape (n_obs,), optional
            Item indices (0 to n_items-1). Required if data is not provided.
        responses : np.ndarray, shape (n_obs,), optional
            Response values. Required if data is not provided.
        person_covariates : pd.DataFrame or np.ndarray, optional
            Person-level covariates (only used if config.hierarchical=True)
        item_covariates : pd.DataFrame or np.ndarray, optional
            Item-level covariates (only used if config.hierarchical=True)
        timepoints : np.ndarray, optional
            Time indices for each observation (only used if config.temporal_dynamics=True)

        inference : {'vi', 'mcmc', 'map'}, default='vi'
            Inference method:
            - 'vi': Variational inference (fast, approximate)
            - 'mcmc': MCMC sampling (slower, exact)
            - 'map': Maximum a posteriori (fastest, point estimate)

        Inference Parameters
        --------------------
        num_samples : int, default=1000
            Number of posterior samples (for VI and MCMC)
        num_warmup : int, default=500
            Number of warmup iterations (for MCMC only)
        num_chains : int, default=1
            Number of parallel chains (for MCMC only)
        vi_steps : int, default=10000
            Number of optimization steps (for VI only)
        vi_optimizer : str, default='adam'
            Optimizer for VI: 'adam', 'sgd', 'adagrad'
        vi_lr : float, default=0.01
            Learning rate for VI
        map_steps : int, default=5000
            Number of optimization steps (for MAP only)
        map_optimizer : str, default='adam'
            Optimizer for MAP: 'adam', 'sgd', 'adagrad'
        map_lr : float, default=0.01
            Learning rate for MAP

        Runtime Parameters
        ------------------
        device : str, default='auto'
            Device to use: 'auto', 'cpu', 'gpu', 'tpu'
        progress_bar : bool, default=True
            Show progress bar during fitting
        random_seed : int, optional
            Random seed for reproducibility

        Advanced
        --------
        chain_method : str, default='parallel'
            MCMC chain execution: 'parallel', 'sequential', 'vectorized'
        guide_type : str, default='normal'
            Variational family for VI: 'normal', 'mvn', 'lowrank_mvn'
        **kwargs
            Additional arguments passed to inference engine

        Returns
        -------
        results : IdealPointResults
            Fitted model results with ideal point estimates and diagnostics

        Examples
        --------
        >>> from idealist import IdealPointEstimator, IdealPointConfig
        >>> from idealist.data import load_data
        >>>
        >>> # Load data
        >>> data = load_data(df, person_col='person', item_col='item', response_col='response')
        >>>
        >>> # Configure model
        >>> config = IdealPointConfig(n_dims=1, response_type='binary')
        >>> estimator = IdealPointEstimator(config)
        >>>
        >>> # Fit with VI (fast)
        >>> results = estimator.fit(data, inference='vi', device='gpu')
        >>>
        >>> # Fit with MCMC (more accurate)
        >>> results = estimator.fit(data, inference='mcmc', num_samples=2000, num_chains=4)
        """
        start_time = time.time()

        # Handle IdealPointData input (convenience)
        from ..data.loaders import IdealPointData
        if isinstance(data, IdealPointData):
            person_ids = data.person_ids
            item_ids = data.item_ids
            responses = data.responses

            # Store names for result interpretation
            self._person_names = data.person_names
            self._item_names = data.item_names

            # Use covariates from data if not explicitly provided
            if person_covariates is None and data.person_covariates is not None:
                person_covariates = data.person_covariates
            if item_covariates is None and data.item_covariates is not None:
                item_covariates = data.item_covariates
        elif isinstance(person_ids, IdealPointData):
            # Backward compatibility: first arg was IdealPointData
            data = person_ids
            person_ids = data.person_ids
            item_ids = data.item_ids
            responses = data.responses
            self._person_names = data.person_names
            self._item_names = data.item_names
            if person_covariates is None and data.person_covariates is not None:
                person_covariates = data.person_covariates
            if item_covariates is None and data.item_covariates is not None:
                item_covariates = data.item_covariates
        else:
            # Direct array input
            self._person_names = None
            self._item_names = None

            # Validate required arrays
            if person_ids is None or item_ids is None or responses is None:
                raise ValueError(
                    "Must provide either 'data' (IdealPointData) or all of person_ids, item_ids, and responses"
                )

        # Store dimensions
        self.config.n_persons = int(person_ids.max() + 1)
        self.config.n_items = int(item_ids.max() + 1)

        # Auto-detect response type if not specified
        if self.config.response_type is None:
            from ..data import detect_response_type
            detected_type, detected_n_categories, detected_bounds = detect_response_type(responses)

            # Update config with detected values
            self.config.response_type = detected_type
            if detected_n_categories is not None and self.config.n_categories is None:
                self.config.n_categories = detected_n_categories
            if detected_bounds is not None and self.config.response_bounds is None:
                self.config.response_bounds = detected_bounds

            # For ordinal responses, remap to 0-indexed if needed
            if detected_type == ResponseType.ORDINAL:
                unique_vals = np.unique(responses)
                if unique_vals[0] != 0:
                    # Remap to 0-indexed
                    val_to_idx = {val: idx for idx, val in enumerate(sorted(unique_vals))}
                    responses = np.array([val_to_idx[val] for val in responses])
                    if progress_bar:
                        logger.info(f"Remapped ordinal responses to 0-indexed (original range: [{unique_vals[0]:.0f}, {unique_vals[-1]:.0f}])")

            # Print detection info
            if progress_bar:
                msg = f"Auto-detected response type: {detected_type.value.upper()}"
                if detected_n_categories is not None:
                    msg += f" (n_categories={detected_n_categories})"
                if detected_bounds is not None:
                    msg += f" (bounds={detected_bounds})"
                logger.info(msg)

        # Auto-select device and parallelization strategy
        if device == 'auto':
            strategy = DeviceManager.auto_select_strategy(
                inference_method=inference,
                n_persons=self.config.n_persons,
                n_items=self.config.n_items,
                n_obs=len(responses),
                use_device='auto',
                max_cpu_chains=4,  # Reasonable default
            )

            # Apply strategy (sets up parallelization)
            DeviceManager.apply_strategy(strategy)

            # Use recommended device
            device = strategy['device']

            # Use recommended num_chains for MCMC
            if inference == 'mcmc' and num_chains == 1 and strategy.get('num_chains'):
                num_chains = strategy['num_chains']
                if progress_bar:
                    logger.info(f"Auto-selected {num_chains} chains for MCMC")

            # Print auto-configuration info
            if progress_bar:
                logger.info(f"Auto-configuration: {strategy['reason']}")

        # Setup device
        use_gpu = (device == 'gpu')
        use_tpu = (device == 'tpu')
        actual_device = DeviceManager.setup_device(
            use_gpu=use_gpu,
            use_tpu=use_tpu,
            distributed=False,  # Can be added later as advanced parameter
        )
        if progress_bar:
            logger.info(f"Using device: {actual_device}")

        if timepoints is not None and self.config.temporal_dynamics:
            self.config.n_timepoints = int(timepoints.max() + 1)

        # Convert to JAX arrays
        person_ids_jax = jnp.array(person_ids, dtype=jnp.int32)
        item_ids_jax = jnp.array(item_ids, dtype=jnp.int32)

        # For ordinal and count responses, use int32; otherwise use default float
        if self.config.response_type in [ResponseType.ORDINAL, ResponseType.COUNT]:
            responses_jax = jnp.array(responses, dtype=jnp.int32)
        else:
            responses_jax = jnp.array(responses)

        person_covariates_jax = jnp.array(person_covariates) if person_covariates is not None else None
        item_covariates_jax = jnp.array(item_covariates) if item_covariates is not None else None
        timepoints_jax = jnp.array(timepoints, dtype=jnp.int32) if timepoints is not None else None

        # Store for predictions
        self.person_ids_train = person_ids_jax
        self.item_ids_train = item_ids_jax
        self.responses_train = responses_jax
        self.person_covariates_train = person_covariates_jax
        self.item_covariates_train = item_covariates_jax
        self.timepoints_train = timepoints_jax

        # Select inference method
        seed = random_seed if random_seed is not None else 0
        rng_key = jax.random.PRNGKey(seed)

        if inference == 'map':
            logger.info("Using MAP inference (point estimation)...")
            results = self._fit_map(
                rng_key, person_ids_jax, item_ids_jax, responses_jax,
                person_covariates_jax, item_covariates_jax, timepoints_jax,
                map_optimizer, map_steps, map_lr, num_samples, progress_bar
            )

        elif inference == 'vi':
            logger.info(f"Using Variational Inference (guide: {guide_type})...")
            results = self._fit_vi(
                rng_key, person_ids_jax, item_ids_jax, responses_jax,
                person_covariates_jax, item_covariates_jax, timepoints_jax,
                guide_type, vi_optimizer, vi_steps, vi_lr, num_samples, progress_bar
            )

        elif inference == 'mcmc':
            logger.info("Using MCMC inference (NUTS)...")
            results = self._fit_mcmc(
                rng_key, person_ids_jax, item_ids_jax, responses_jax,
                person_covariates_jax, item_covariates_jax, timepoints_jax,
                num_warmup, num_samples, num_chains, chain_method, progress_bar, **kwargs
            )

        else:
            raise ValueError(f"Unknown inference method: {inference}. Use 'map', 'vi', or 'mcmc'.")

        # Compute results
        computation_time = time.time() - start_time

        results_obj = self._compute_results(
            self.posterior_samples,
            computation_time,
            inference
        )

        # Store original names in results if available
        results_obj._person_names = self._person_names
        results_obj._item_names = self._item_names

        self.results = results_obj
        self._is_fitted = True
        return results_obj

    def _fit_map(
        self,
        rng_key,
        person_ids, item_ids, responses,
        person_covariates, item_covariates, timepoints,
        optimizer, steps, lr, num_samples, progress_bar
    ):
        """MAP estimation via SVI with point mass guide."""
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoDelta
        from numpyro.optim import Adam, SGD, Adagrad

        # Point mass guide (delta functions at MAP estimate)
        guide = AutoDelta(self._build_model)

        # Select optimizer
        if optimizer == 'adam':
            opt = Adam(lr)
        elif optimizer == 'sgd':
            opt = SGD(lr)
        elif optimizer == 'adagrad':
            opt = Adagrad(lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # SVI
        svi = SVI(self._build_model, guide, opt, Trace_ELBO())

        # Run optimization
        svi_result = svi.run(
            rng_key, steps,
            person_ids=person_ids,
            item_ids=item_ids,
            responses=responses,
            person_covariates=person_covariates,
            item_covariates=item_covariates,
            timepoints=timepoints,
            progress_bar=progress_bar
        )

        self.svi = svi
        self.guide = guide

        # Get MAP estimates
        params = svi_result.params

        # Generate samples from guide (will be point masses)
        predictive = Predictive(guide, params=params, num_samples=num_samples)
        samples_key = jax.random.PRNGKey(1)
        self.posterior_samples = predictive(
            samples_key,
            person_ids=person_ids,
            item_ids=item_ids,
            responses=responses,
            person_covariates=person_covariates,
            item_covariates=item_covariates,
            timepoints=timepoints
        )

        return svi_result

    def _fit_vi(
        self,
        rng_key,
        person_ids, item_ids, responses,
        person_covariates, item_covariates, timepoints,
        guide_type, optimizer, steps, lr, num_samples, progress_bar
    ):
        """Variational inference."""
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.optim import Adam, SGD, Adagrad

        # Select guide
        if guide_type == 'normal':
            guide = AutoNormal(self._build_model)
        elif guide_type == 'mvn':
            guide = AutoMultivariateNormal(self._build_model)
        elif guide_type == 'lowrank_mvn':
            guide = AutoLowRankMultivariateNormal(self._build_model)
        else:
            raise ValueError(f"Unknown guide type: {guide_type}")

        # Select optimizer
        if optimizer == 'adam':
            opt = Adam(lr)
        elif optimizer == 'sgd':
            opt = SGD(lr)
        elif optimizer == 'adagrad':
            opt = Adagrad(lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # SVI
        svi = SVI(self._build_model, guide, opt, Trace_ELBO())

        # Run optimization
        svi_result = svi.run(
            rng_key, steps,
            person_ids=person_ids,
            item_ids=item_ids,
            responses=responses,
            person_covariates=person_covariates,
            item_covariates=item_covariates,
            timepoints=timepoints,
            progress_bar=progress_bar
        )

        self.svi = svi
        self.guide = guide

        # Sample from approximate posterior
        predictive = Predictive(guide, params=svi_result.params, num_samples=num_samples)
        samples_key = jax.random.PRNGKey(1)
        self.posterior_samples = predictive(
            samples_key,
            person_ids=person_ids,
            item_ids=item_ids,
            responses=responses,
            person_covariates=person_covariates,
            item_covariates=item_covariates,
            timepoints=timepoints
        )

        return svi_result

    def _fit_mcmc(
        self,
        rng_key,
        person_ids, item_ids, responses,
        person_covariates, item_covariates, timepoints,
        num_warmup, num_samples, num_chains, chain_method, progress_bar,
        **kwargs
    ):
        """MCMC sampling via NUTS."""
        nuts_kernel = NUTS(
            self._build_model,
            target_accept_prob=0.8,
            max_tree_depth=10,
        )

        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=progress_bar,
            **kwargs
        )

        mcmc.run(
            rng_key,
            person_ids=person_ids,
            item_ids=item_ids,
            responses=responses,
            person_covariates=person_covariates,
            item_covariates=item_covariates,
            timepoints=timepoints,
        )

        self.mcmc = mcmc

        # Get samples grouped by chain for per-chain identification constraint
        samples_by_chain = mcmc.get_samples(group_by_chain=True)

        # Apply identification constraint per-chain to avoid label switching
        # across chains (each chain may explore different reflection modes)
        self.posterior_samples = self._apply_per_chain_identification(samples_by_chain)

        return mcmc

    def _apply_per_chain_identification(self, samples_by_chain: Dict) -> Dict:
        """
        Apply identification constraint per-chain before concatenating.

        Ideal point models have reflection invariance: (θ, β) and (-θ, -β) produce
        identical likelihoods. When running multiple MCMC chains, different
        chains may explore different modes (positive vs negative reflection).
        This causes label switching across chains, leading to poor convergence
        diagnostics and biased estimates.

        Solution: Apply identification constraint to each chain independently
        before concatenating them.

        Args:
            samples_by_chain: Dict of samples with shape (num_chains, num_samples, ...)

        Returns:
            Dict of concatenated samples with shape (num_chains * num_samples, ...)
            after per-chain identification.
        """
        import jax.numpy as jnp

        # Check if we have multiple chains
        first_param = next(iter(samples_by_chain.values()))
        if len(first_param.shape) < 2:
            # Only one chain or already concatenated, return as-is
            return samples_by_chain

        num_chains = first_param.shape[0]

        # Identify which parameters need sign flipping
        theta_key = "ideal_points" if "ideal_points" in samples_by_chain else "theta_initial"
        disc_key = "discrimination"

        # Storage for fixed chains
        fixed_chains = {key: [] for key in samples_by_chain.keys()}

        # Process each chain independently
        for chain_idx in range(num_chains):
            # Extract this chain's samples
            chain_samples = {
                key: samples_by_chain[key][chain_idx]
                for key in samples_by_chain.keys()
            }

            # Check sign using discrimination sum (more stable than mean)
            disc_samples = chain_samples[disc_key]
            sum_disc = jnp.sum(disc_samples[:, :, 0])  # Sum over samples and items, first dim

            # Apply identification constraint if needed
            if sum_disc < 0:
                # Flip ideal points and discrimination
                chain_samples[theta_key] = -chain_samples[theta_key]
                chain_samples[disc_key] = -chain_samples[disc_key]

                # For temporal models, also flip innovations
                if "theta_innovations" in chain_samples:
                    chain_samples["theta_innovations"] = -chain_samples["theta_innovations"]

            # Store fixed chain
            for key in chain_samples.keys():
                fixed_chains[key].append(chain_samples[key])

        # Concatenate all fixed chains along the sample dimension
        concatenated = {}
        for key in fixed_chains.keys():
            # Stack chains: (num_chains, num_samples, ...) then reshape to (num_chains * num_samples, ...)
            stacked = jnp.stack(fixed_chains[key], axis=0)
            # Concatenate along chain dimension
            concatenated[key] = jnp.concatenate(
                [stacked[i] for i in range(num_chains)],
                axis=0
            )

        return concatenated

    def _compute_results(
        self,
        posterior_samples: Dict,
        computation_time: float,
        inference_method: str,
    ) -> IdealPointResults:
        """Extract results from posterior samples."""

        # Extract ideal points
        temporal_ideal_points_mean = None

        if "ideal_points" in posterior_samples:
            ideal_points_samples = posterior_samples["ideal_points"]
        elif "theta_initial" in posterior_samples:
            # Temporal model
            theta_0_samples = posterior_samples["theta_initial"]
            if "theta_innovations" in posterior_samples:
                innovations_samples = posterior_samples["theta_innovations"]
                cumulative_innovations = jnp.cumsum(innovations_samples, axis=1)
                temporal_samples = jnp.concatenate([
                    theta_0_samples[:, None, :, :],
                    theta_0_samples[:, None, :, :] + cumulative_innovations
                ], axis=1)
                temporal_ideal_points_mean = np.array(jnp.mean(temporal_samples, axis=0))
            ideal_points_samples = theta_0_samples
        else:
            raise ValueError("No ideal points found in posterior samples")

        # Extract item parameters
        # Note: Identification constraint has already been applied per-chain
        # in _apply_per_chain_identification() before samples reach here
        difficulty_samples = posterior_samples["difficulty"]
        discrimination_samples = posterior_samples["discrimination"]

        # Compute summaries
        ideal_points_mean = np.array(jnp.mean(ideal_points_samples, axis=0))
        ideal_points_std = np.array(jnp.std(ideal_points_samples, axis=0))
        ideal_points_ci_lower = np.array(jnp.percentile(ideal_points_samples, 2.5, axis=0))
        ideal_points_ci_upper = np.array(jnp.percentile(ideal_points_samples, 97.5, axis=0))

        difficulty_mean = np.array(jnp.mean(difficulty_samples, axis=0))
        difficulty_std = np.array(jnp.std(difficulty_samples, axis=0))

        discrimination_mean = np.array(jnp.mean(discrimination_samples, axis=0))
        discrimination_std = np.array(jnp.std(discrimination_samples, axis=0))

        # Extract covariate effects if present
        person_covariate_effects = None
        if "person_covariate_effects" in posterior_samples:
            person_covariate_effects = np.array(jnp.mean(posterior_samples["person_covariate_effects"], axis=0))

        item_covariate_effects = None
        if "item_difficulty_covariate_effects" in posterior_samples or "item_discrimination_covariate_effects" in posterior_samples:
            item_covariate_effects = {}
            if "item_difficulty_covariate_effects" in posterior_samples:
                item_covariate_effects['difficulty'] = np.array(jnp.mean(posterior_samples["item_difficulty_covariate_effects"], axis=0))
            if "item_discrimination_covariate_effects" in posterior_samples:
                item_covariate_effects['discrimination'] = np.array(jnp.mean(posterior_samples["item_discrimination_covariate_effects"], axis=0))

        # Convergence info
        convergence_info = {
            "method": inference_method.upper(),
            "num_samples": len(posterior_samples[list(posterior_samples.keys())[0]]),
        }

        if self.mcmc is not None:
            # Compute comprehensive MCMC diagnostics
            mcmc_diagnostics = self._compute_mcmc_diagnostics(posterior_samples)

            convergence_info.update({
                "num_chains": self.mcmc.num_chains,
                "num_warmup": self.mcmc.num_warmup,
                "mcmc_diagnostics": mcmc_diagnostics,
            })

        # Create results
        results = IdealPointResults(
            ideal_points=ideal_points_mean,
            ideal_points_std=ideal_points_std,
            ideal_points_ci_lower=ideal_points_ci_lower,
            ideal_points_ci_upper=ideal_points_ci_upper,
            ideal_points_samples=np.array(ideal_points_samples),
            difficulty=difficulty_mean,
            difficulty_std=difficulty_std,
            discrimination=discrimination_mean,
            discrimination_std=discrimination_std,
            temporal_ideal_points=temporal_ideal_points_mean,
            person_covariate_effects=person_covariate_effects,
            item_covariate_effects=item_covariate_effects,
            convergence_info=convergence_info,
            computation_time=computation_time,
            log_likelihood=None,
        )

        return results

    def _compute_mcmc_diagnostics(self, posterior_samples: Dict) -> Dict:
        """
        Compute comprehensive MCMC diagnostics.

        Includes:
        - Effective Sample Size (ESS) bulk and tail
        - Split R-hat (potential scale reduction factor)
        - NUTS acceptance probability
        - Number of divergences
        - Tree depth statistics
        - Energy diagnostics

        Parameters
        ----------
        posterior_samples : Dict
            Posterior samples from MCMC

        Returns
        -------
        Dict : Diagnostic statistics
        """
        from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

        diagnostics = {}

        # Get MCMC extra fields (NUTS diagnostics)
        if hasattr(self.mcmc, 'get_extra_fields'):
            extra_fields = self.mcmc.get_extra_fields()

            # NUTS acceptance probability
            if 'accept_prob' in extra_fields:
                accept_prob = extra_fields['accept_prob']
                diagnostics['acceptance_rate'] = {
                    'mean': float(np.mean(accept_prob)),
                    'min': float(np.min(accept_prob)),
                    'max': float(np.max(accept_prob)),
                    'std': float(np.std(accept_prob)),
                }

            # Divergences
            if 'diverging' in extra_fields:
                n_divergences = int(np.sum(extra_fields['diverging']))
                total_samples = extra_fields['diverging'].size
                diagnostics['divergences'] = {
                    'n_divergences': n_divergences,
                    'total_samples': total_samples,
                    'divergence_rate': float(n_divergences / total_samples),
                }

            # Tree depth
            if 'tree_depth' in extra_fields:
                tree_depth = extra_fields['tree_depth']
                diagnostics['tree_depth'] = {
                    'mean': float(np.mean(tree_depth)),
                    'max': int(np.max(tree_depth)),
                    'max_treedepth_reached': int(np.sum(tree_depth >= 10)),  # Default max is 10
                }

            # Energy diagnostics (if available)
            if 'energy' in extra_fields:
                energy = extra_fields['energy']
                diagnostics['energy'] = {
                    'mean': float(np.mean(energy)),
                    'std': float(np.std(energy)),
                }

        # Compute ESS and Rhat for key parameters
        param_diagnostics = {}

        for param_name in ['ideal_points', 'difficulty', 'discrimination',
                           'theta_initial', 'theta_innovations']:
            if param_name not in posterior_samples:
                continue

            samples = posterior_samples[param_name]

            # Convert to numpy if JAX array
            if hasattr(samples, 'copy'):
                samples = np.array(samples)

            # Reshape for ESS/Rhat computation
            # Expected shape: (n_chains, n_samples_per_chain, ...)
            if samples.ndim >= 2:
                # For NumPyro MCMC with multiple chains, samples are already
                # structured as (n_chains * n_samples_per_chain, ...)
                # We need to reshape to (n_chains, n_samples_per_chain, ...)

                num_chains = self.mcmc.num_chains
                total_samples = samples.shape[0]
                samples_per_chain = total_samples // num_chains

                if total_samples % num_chains == 0:
                    # Reshape to (chains, samples_per_chain, ...)
                    new_shape = (num_chains, samples_per_chain) + samples.shape[1:]
                    samples_reshaped = samples.reshape(new_shape)

                    # Compute ESS
                    try:
                        ess_bulk = effective_sample_size(samples_reshaped)
                        ess_bulk_min = float(np.min(ess_bulk))
                        ess_bulk_mean = float(np.mean(ess_bulk))
                    except Exception as e:
                        ess_bulk_min = None
                        ess_bulk_mean = None

                    # Compute Rhat
                    try:
                        rhat = split_gelman_rubin(samples_reshaped)
                        rhat_max = float(np.max(rhat))
                        rhat_mean = float(np.mean(rhat))
                    except Exception as e:
                        rhat_max = None
                        rhat_mean = None

                    param_diagnostics[param_name] = {
                        'ess_bulk_min': ess_bulk_min,
                        'ess_bulk_mean': ess_bulk_mean,
                        'rhat_max': rhat_max,
                        'rhat_mean': rhat_mean,
                        'shape': samples.shape,
                    }

        diagnostics['parameters'] = param_diagnostics

        # Summary statistics
        if param_diagnostics:
            all_ess_mins = [v['ess_bulk_min'] for v in param_diagnostics.values()
                            if v['ess_bulk_min'] is not None]
            all_rhat_maxes = [v['rhat_max'] for v in param_diagnostics.values()
                              if v['rhat_max'] is not None]

            if all_ess_mins:
                diagnostics['summary'] = {
                    'min_ess': float(np.min(all_ess_mins)),
                    'max_rhat': float(np.max(all_rhat_maxes)) if all_rhat_maxes else None,
                    'all_rhat_below_1_1': all(r < 1.1 for r in all_rhat_maxes) if all_rhat_maxes else None,
                    'all_ess_above_400': all(e > 400 for e in all_ess_mins),
                }

        return diagnostics

    def print_mcmc_diagnostics(self):
        """
        Print MCMC diagnostics in human-readable format.

        Requires that model was fit with inference='mcmc' and diagnostics are available.
        """
        if not hasattr(self, 'results') or self.results is None:
            print("No results available. Please fit the model first.")
            return

        conv_info = self.results.convergence_info

        if 'mcmc_diagnostics' not in conv_info:
            print("MCMC diagnostics not available.")
            print("Model may not have been fit with MCMC inference.")
            return

        diag = conv_info['mcmc_diagnostics']

        print("=" * 80)
        print("MCMC DIAGNOSTICS")
        print("=" * 80)
        print()

        # Basic info
        print(f"Chains: {conv_info.get('num_chains', 'N/A')}")
        print(f"Warmup samples: {conv_info.get('num_warmup', 'N/A')}")
        print(f"Post-warmup samples: {conv_info.get('num_samples', 'N/A')}")
        print()

        # Acceptance rate
        if 'acceptance_rate' in diag:
            acc = diag['acceptance_rate']
            print("NUTS Acceptance Probability:")
            print(f"  Mean: {acc['mean']:.3f}")
            print(f"  Range: [{acc['min']:.3f}, {acc['max']:.3f}]")

            # Interpret
            if acc['mean'] < 0.6:
                print("  WARNING: Low acceptance rate! Consider increasing adapt_delta.")
            elif acc['mean'] > 0.95:
                print("  NOTE: Very high acceptance rate. Could increase efficiency.")
            else:
                print("  STATUS: Good acceptance rate.")
            print()

        # Divergences
        if 'divergences' in diag:
            div = diag['divergences']
            print(f"Divergences:")
            print(f"  Count: {div['n_divergences']} / {div['total_samples']}")
            print(f"  Rate: {div['divergence_rate']:.2%}")

            if div['n_divergences'] > 0:
                print("  WARNING: Divergences detected! Results may be unreliable.")
                print("  Consider:")
                print("    - Increasing adapt_delta (e.g., target_accept_prob=0.9)")
                print("    - Reparameterizing the model")
                print("    - Checking for poor identifiability")
            else:
                print("  STATUS: No divergences detected.")
            print()

        # Tree depth
        if 'tree_depth' in diag:
            tree = diag['tree_depth']
            print(f"Tree Depth:")
            print(f"  Mean: {tree['mean']:.1f}")
            print(f"  Max: {tree['max']}")

            if tree['max_treedepth_reached'] > 0:
                pct = tree['max_treedepth_reached'] / diag.get('divergences', {}).get('total_samples', 1)
                print(f"  Max tree depth reached: {tree['max_treedepth_reached']} times ({pct:.1%})")
                print("  NOTE: Consider increasing max_tree_depth if this is frequent.")
            print()

        # Parameter-specific diagnostics
        if 'parameters' in diag and diag['parameters']:
            print("=" * 80)
            print("PARAMETER DIAGNOSTICS")
            print("=" * 80)
            print()

            print(f"{'Parameter':<25} {'Min ESS':>10} {'Mean ESS':>10} {'Max Rhat':>10} {'Mean Rhat':>10}")
            print("-" * 80)

            for param_name, param_diag in diag['parameters'].items():
                ess_min = param_diag.get('ess_bulk_min')
                ess_mean = param_diag.get('ess_bulk_mean')
                rhat_max = param_diag.get('rhat_max')
                rhat_mean = param_diag.get('rhat_mean')

                ess_min_str = f"{ess_min:10.1f}" if ess_min is not None else "       N/A"
                ess_mean_str = f"{ess_mean:10.1f}" if ess_mean is not None else "       N/A"
                rhat_max_str = f"{rhat_max:10.4f}" if rhat_max is not None else "       N/A"
                rhat_mean_str = f"{rhat_mean:10.4f}" if rhat_mean is not None else "       N/A"

                print(f"{param_name:<25} {ess_min_str} {ess_mean_str} {rhat_max_str} {rhat_mean_str}")

            print()

        # Summary assessment
        if 'summary' in diag:
            summ = diag['summary']
            print("=" * 80)
            print("OVERALL ASSESSMENT")
            print("=" * 80)
            print()

            print(f"Minimum ESS (across all parameters): {summ['min_ess']:.1f}")

            if summ['min_ess'] < 100:
                print("  WARNING: Very low ESS! Need more samples or better mixing.")
            elif summ['min_ess'] < 400:
                print("  CAUTION: Low ESS. Consider running longer.")
            else:
                print("  STATUS: Adequate ESS.")
            print()

            if summ['max_rhat'] is not None:
                print(f"Maximum Rhat (across all parameters): {summ['max_rhat']:.4f}")

                if summ['all_rhat_below_1_1']:
                    print("  STATUS: All Rhat < 1.1 (convergence criteria met).")
                else:
                    print("  WARNING: Some Rhat >= 1.1 (convergence issues!).")
                    print("  Consider:")
                    print("    - Running more warmup iterations")
                    print("    - Running more chains")
                    print("    - Checking for multimodality")
            print()

            # Overall verdict
            print("VERDICT:")
            if (summ['min_ess'] >= 400 and
                summ.get('all_rhat_below_1_1', False) and
                diag.get('divergences', {}).get('n_divergences', 0) == 0):
                print("  PASS: MCMC diagnostics look good!")
            elif (summ['min_ess'] >= 200 and
                  summ.get('max_rhat', 1.1) < 1.15):
                print("  ACCEPTABLE: MCMC diagnostics are reasonable.")
                print("  Consider running longer for publication-quality results.")
            else:
                print("  FAIL: MCMC diagnostics indicate problems!")
                print("  Results may be unreliable. Address issues above.")
            print()

        print("=" * 80)

    def predict(
        self,
        person_ids: np.ndarray,
        item_ids: np.ndarray,
        timepoints: Optional[np.ndarray] = None,
        return_samples: bool = False,
    ) -> Union[np.ndarray, Dict]:
        """
        Predict responses for person-item pairs.

        Parameters
        ----------
        person_ids : np.ndarray
            Person indices
        item_ids : np.ndarray
            Item indices
        timepoints : Optional[np.ndarray]
            Time indices for temporal models
        return_samples : bool
            If True, return full posterior predictive samples

        Returns
        -------
        predictions : np.ndarray or Dict
            If return_samples=False: posterior mean predictions
            If return_samples=True: dict with 'mean', 'std', 'samples'
        """
        if self.posterior_samples is None:
            raise ValueError("Model must be fitted first")

        person_ids_jax = jnp.array(person_ids, dtype=jnp.int32)
        item_ids_jax = jnp.array(item_ids, dtype=jnp.int32)
        timepoints_jax = jnp.array(timepoints, dtype=jnp.int32) if timepoints is not None else None

        # Generate posterior predictive
        predictive = Predictive(self._build_model, posterior_samples=self.posterior_samples)
        rng_key = jax.random.PRNGKey(1)

        predictions = predictive(
            rng_key,
            person_ids=person_ids_jax,
            item_ids=item_ids_jax,
            person_covariates=self.person_covariates_train,
            item_covariates=self.item_covariates_train,
            timepoints=timepoints_jax,
        )

        pred_samples = predictions["obs"]

        if return_samples:
            return {
                "mean": np.array(jnp.mean(pred_samples, axis=0)),
                "std": np.array(jnp.std(pred_samples, axis=0)),
                "samples": np.array(pred_samples),
            }
        else:
            return np.array(jnp.mean(pred_samples, axis=0))

    def save(self, filepath: str):
        """
        Save fitted model to disk.

        This method overrides the base class to avoid pickling unpicklable
        optimizer state (SVI closures). Only essential model state is saved.

        Parameters
        ----------
        filepath : str
            Path to save file (.pkl recommended)
        """
        import pickle
        from dataclasses import asdict

        if not self._is_fitted:
            raise ValueError("Model must be fitted before saving")

        # Create save dictionary with only picklable components
        save_dict = {
            'config': asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else self.config.__dict__,
            'posterior_samples': {
                k: np.array(v) for k, v in self.posterior_samples.items()
            },
            'results': asdict(self.results) if self.results and hasattr(self.results, '__dataclass_fields__') else None,
            'person_ids_train': np.array(self.person_ids_train) if self.person_ids_train is not None else None,
            'item_ids_train': np.array(self.item_ids_train) if self.item_ids_train is not None else None,
            'responses_train': np.array(self.responses_train) if self.responses_train is not None else None,
            'person_covariates_train': np.array(self.person_covariates_train) if self.person_covariates_train is not None else None,
            'item_covariates_train': np.array(self.item_covariates_train) if self.item_covariates_train is not None else None,
            'timepoints_train': np.array(self.timepoints_train) if self.timepoints_train is not None else None,
        }

        # Do NOT save self.svi, self.guide, self.mcmc - they contain unpicklable closures

        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str) -> 'IdealPointEstimator':
        """
        Load fitted model from disk.

        Parameters
        ----------
        filepath : str
            Path to saved file

        Returns
        -------
        model : IdealPointEstimator
            Loaded fitted model
        """
        import pickle
        from dataclasses import fields

        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)

        # Reconstruct config
        config_dict = save_dict['config']
        # Convert enum strings back to enums if needed
        from ..core.base import ResponseType, IdentificationConstraint
        if 'response_type' in config_dict and isinstance(config_dict['response_type'], str):
            config_dict['response_type'] = ResponseType(config_dict['response_type'])
        if 'identification' in config_dict and isinstance(config_dict['identification'], str):
            config_dict['identification'] = IdentificationConstraint(config_dict['identification'])

        config = IdealPointConfig(**config_dict)

        # Create model instance
        model = cls(config)

        # Restore state
        model.posterior_samples = {
            k: jnp.array(v) for k, v in save_dict['posterior_samples'].items()
        }

        # Restore results
        if save_dict['results'] is not None:
            model.results = IdealPointResults(**save_dict['results'])

        # Restore training data
        model.person_ids_train = jnp.array(save_dict['person_ids_train']) if save_dict['person_ids_train'] is not None else None
        model.item_ids_train = jnp.array(save_dict['item_ids_train']) if save_dict['item_ids_train'] is not None else None
        model.responses_train = jnp.array(save_dict['responses_train']) if save_dict['responses_train'] is not None else None
        model.person_covariates_train = jnp.array(save_dict['person_covariates_train']) if save_dict['person_covariates_train'] is not None else None
        model.item_covariates_train = jnp.array(save_dict['item_covariates_train']) if save_dict['item_covariates_train'] is not None else None
        model.timepoints_train = jnp.array(save_dict['timepoints_train']) if save_dict['timepoints_train'] is not None else None

        model._is_fitted = True

        return model
