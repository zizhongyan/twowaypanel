====================
Quick Start Tutorial
====================

This page introduces the core functionality of **twowaypanel** and explains the package’s main entry point: 
:func:`twowaypanel.fit`. 


Background and scope
--------------------

``twowaypanel`` implements the likelihood-based bias-correction methods proposed in Yan et al. (2026), including:

- **Integrated-likelihood-based correction** (via *priors*; “prior correction”), and
- **Joint-likelihood-based correction** (via *penalties*; “penalty correction”).

For logit and probit panels, the package also provides the **analytical bias correction** for fixed-effects MLE
developed by Fernández-Val and Weidner (2016).



Supported models and methods
----------------------------

The table below summarizes the model classes and bias-correction methods currently supported by the package.

.. list-table::
   :header-rows: 1
   :widths: 22 18 18 26 18

   * - Model class
     - Regressor exogeneity
     - Generic prior / penalty
     - Model-specific prior / penalty
     - Analytical correction (FW16)
   * - Binary logit
     - Strict exogenous
     - ✓
     - ✓ (Prior SE)
     - ✓
   * - Binary logit
     - Predetermined
     - ✓
     - ✓ (Prior PE; Prior SE + analytical)
     - ✓
   * - Binary Probit
     - Strict exog. or predet.
     - ✓
     - –
     - ✓
   * - Multinomial logit
     - Strict exogenous
     - ✓
     - ✓ (Prior SML)
     - –
   * - Multinomial logit
     - Predetermined
     - ✓
     - ✓ (Prior PML)
     - –
   * - Ordered logit
     - Strict exog. or predet.
     - ✓
     - –
     - –

**Notes.**

- *Model-specific* priors/penalties are defined in **Yan et al. (2026)** and are designed to deliver improved
  finite-sample performance within the corresponding model class.

- The *generic* prior/penalty is intended to be **robust across a broad class of nonlinear panel specifications**;
  see **Yan et al. (2026)** for details and motivation.

- The package can also compute **average partial effects (APEs)**. Both **continuous-regressor** APEs and **discrete-regressor** APEs (finite changes) are supported.


The main entry point: ``twowaypanel.fit``
-----------------------------------------

Most users will interact with the package through a single function:

.. code-block:: python

   twowaypanel.fit(
       Y, X=None, model=None,
       prior=None, lag=0, ac=False,
       algorithm="JML",
       X_names=None, sv=None, silent=False, ape=True, cutoff0=0,
       mcmc_iters=16000, mcmc_burnin=1000, mcmc_skipsize=2, mcmc_timer=1, 
       mcmc_sv_mle=True, mcmc_diagnosis=True,
       beta_variance=None, fe_variance=0.3, block_size=8
   )


Inputs and arguments
--------------------

Conceptually, :func:`twowaypanel.fit` takes panel data arrays (``Y`` and ``X``), a model choice (``model``),
and optional bias-correction and inference settings (``prior``, ``ac``, ``lag``, etc.), and returns an estimation
result object (and optionally prints a summary table).

Inputs: data arrays
~~~~~~~~~~~~~~~~~~~

**Dependent variable (Y).**
``Y`` must be provided and should be arranged as an ``N × T`` object (typically a 2D NumPy array), where
``N`` is the number of individuals and ``T`` is the number of time periods.

**Regressors (X).**
``X`` must be provided and should be arranged as an ``N × T × K`` 3D NumPy array, where ``K`` is the number of
covariates. The ordering is:

- first index: individual ``i = 1, …, N``,
- second index: time ``t = 1, …, T``,
- third index: covariate ``k = 1, …, K``.

Variable names (optional).
``X_names`` is optional and is used only for labeling output tables. It should be a Python list of length ``K`` (e.g., ``["x1", "x2", ...]``).
If omitted, the package uses default labels ``X_1, X_2, …``.

Model selection
~~~~~~~~~~~~~~~

The argument ``model`` is required and selects the likelihood:

- ``model="logit"``  : binary logit
- ``model="probit"`` : binary probit
- ``model="ologit"`` : ordered logit
- ``model="mlogit"`` : multinomial logit

Bias correction: ``prior`` and ``ac``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Likelihood-based correction (``prior``).
The argument ``prior`` selects which prior/penalty is used for bias correction. We use a single keyword
``prior`` to control both “prior correction” and “penalty correction”; the interpretation depends on the
estimation algorithm (see ``algorithm`` below).

- If ``prior`` is omitted (or set to ``None``), no likelihood-based correction is applied.
- Common choices (following Yan et al., 2026) include:

  - ``prior="Generic"`` : generic correction for a broad class of nonlinear panels
  - ``prior="SE"``      : for binary logit (with **strictly exogenous regressors**)
  - ``prior="PE"``      : for binary logit (with **predetermined regressors**)
  - ``prior="SML"``     : for multinomial logit (with **strictly exogenous regressors**)
  - ``prior="PML"``     : for multinomial logit (with **predetermined regressors**)

Analytical correction (``ac``).
The option ``ac`` (default ``False``) enables the analytical bias correction for fixed-effects MLE. It applies to
**binary logit** and **binary probit** models.

- If ``ac=True`` and no prior is specified, the package applies the Fernández-Val and Weidner (2016) correction to
  the fixed-effects MLE.

- In the logit setting with predetermined regressors, if ``prior="SE"`` and ``ac=True``, the workflow follows Yan et al. (2026): the
  likelihood is partially corrected via the prior, and remaining cross-time dependence components are handled via
  an additional analytical correction step after estimation.

Serial dependence / trimming: ``lag``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``lag=0``, all regressors are treated as strictly exogenous.

In settings with predetermined regressors, several correction terms involve **spectral expectations**. In practice, these objects are approximated using a **truncated lag window**. The argument ``lag`` is a trimming parameter that controls the truncation level.

- Larger ``lag`` allows richer serial dependence to enter the approximation.
- The same ``lag`` parameter is used when truncation is required by the analytical correction step.

As a rule of thumb, ``lag`` should be chosen in line with the persistence in the data.
See the Examples and the replication files for practical choices.

Estimation algorithm: ``algorithm``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The argument ``algorithm`` selects how common parameters and fixed effects are estimated:

- ``algorithm="JML"`` (default): joint maximum likelihood estimation.
  In this case, a “prior correction” is implemented as a **penalty term** added to the likelihood.

- ``algorithm="MCMC"``: Metropolis–Hastings sampling.
  This option is useful when you want full posterior-style uncertainty quantification under the prior-based
  formulation; computational details are described in Yan et al. (2026).

Output control: ``silent`` and APEs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``silent`` (default ``False``): if set to ``True``, the function will not print the summary table to screen.
  The fitted results are still returned.

- ``ape`` (default ``True``): if set to ``True``, the function computes APEs in addition to coefficient estimates.

Ordered logit identification: ``cutoff0``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ordered logit models, one cutoff must be normalized for identification. The option ``cutoff0`` sets the fixed
value for the first cutoff (default ``0``).

Starting values: ``sv``
~~~~~~~~~~~~~~~~~~~~~~~

``sv`` allows users to provide starting values for optimization. In most applications, the default behavior is
sufficient and manual tuning is not required.

MCMC tuning parameters (only when ``algorithm="MCMC"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The remaining parameters affect MCMC computation and are only relevant when ``algorithm="MCMC"``:

- ``mcmc_iters``: number of MCMC iterations (default ``16000``)
- ``mcmc_burnin``: burn-in iterations (default ``1000``)
- ``mcmc_skipsize``: keep one draw every ``mcmc_skipsize`` iterations (default ``2``)
- ``mcmc_timer``: a progress bar to display frequency (default ``1``)

Initialization:

- ``mcmc_sv_mle``: whether to initialize MCMC at MLE-based starting values;
  if set to ``True``, the chain starts from MLE-based initial values, which is often helpful in practice.

Diagnostics:

- ``mcmc_diagnosis``: if enabled, produces convergence diagnostics (e.g., Geweke tests) and common trace/histogram/ACF
  plots for key parameters.

Proposal distribution (Metropolis–Hastings):

- ``beta_variance`` and ``fe_variance``: variances of Gaussian random-walk proposals in early iterations (the first
  400 iterations); afterwards, a self-adaptive scheme is used.
- ``block_size``: number of parameters updated per block in blockwise proposals (default ``8``).


Returned object and key attributes
----------------------------------

:func:`twowaypanel.fit` returns a **result object** that stores parameter estimates, standard errors, and (optionally)
average partial effects (APEs). The most commonly used attributes are:

Main outputs
^^^^^^^^^^^^

Below outputs are available under ``algorithm="JML"`` and ``algorithm="MCMC"``.

- ``self.paras``:
  Estimated **common parameters** (e.g., regression coefficients). In typical use, this corresponds to the parameter
  vector :math:`\beta`.

- ``self.se``:
  Asymptotic **standard errors** for the common parameters in ``self.paras``.

- ``self.feparas``:
  Estimated **fixed effects parameters**, including individual effects and time effects (two-way fixed effects).

- ``self.success``:
  A success indicator for the estimation routine. ``1`` indicates successful convergence/termination; otherwise the
  optimization or sampling did not complete as intended.

- ``self.fun``:
  The final value of the objective function at the solution. Under ``algorithm="JML"``, this is the final
  (penalized) log-likelihood value used by the optimizer.

- ``self.ape``:
  Estimated **average partial effects (APEs)** when ``ape=True``. For discrete regressors, APEs are computed as finite
  changes; for continuous regressors, APEs correspond to average marginal effects.

- ``self.apese``:
  Asymptotic **standard errors** for the APE estimates in ``self.ape``.

MCMC-specific outputs (only when ``algorithm="MCMC"``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``algorithm="MCMC"``, the result object additionally stores posterior samples and MCMC diagnostics:

- ``self.theta_mcmc``:
  MCMC draws for the **common parameters** (e.g., :math:`\beta`), i.e., a matrix/array of sampled values across
  iterations.

- ``self.lambda_mcmc``:
  MCMC draws for the **fixed effects parameters** (individual and time effects), i.e., sampled values across
  iterations.

- ``self.sd``:
  Monte Carlo standard deviations of the MCMC draws for each parameter (computed from the retained samples).

- ``self.accRateTheta``:
  Acceptance rates for the Metropolis–Hastings updates of the **common parameters**.

- ``self.accRateFE``:
  Acceptance rates for the Metropolis–Hastings updates of the **fixed effects parameters**.



Where to go next
----------------

- For worked examples by model class (binary logit/probit/multinomial/ordered), see the **Examples** section.
- For paper-specific replication scripts (Monte Carlo and empirical illustration), see the **Replication** materials
  referenced in the repository.

