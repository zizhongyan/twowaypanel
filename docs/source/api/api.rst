=============
API Reference
=============

This page documents the main public-facing functions shipped with **twowaypanel**.
The focus is on (i) data loaders and generators used in the examples and replication
materials, and (ii) the estimation interface.

For most applied users, the central function is :func:`twowaypanel.fit`. Since it is
introduced in detail in the *Quick Start Tutorial*, we provide only a concise technical
summary here and refer readers to the tutorial for intuition and recommended workflows.

.. contents::
   :local:
   :depth: 2


Data generation for examples and testing
----------------------------------------

``twowaypanel.demo.PanelGenData``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: twowaypanel.demo.PanelGenData(N, T, seed="2025", model="logit", dynamic=0)

   Generate an artificial panel dataset for nonlinear models with two-way fixed effects.

   This helper is primarily intended for:
   (i) quick demonstrations in the Examples section,
   (ii) unit tests / sanity checks, and
   (iii) Monte Carlo experiments.

   Parameters
   ----------
   N : int
       Number of individuals.

   T : int
       Number of time periods.

   seed : str or int, default="2025"
       Random seed used for reproducibility.

   model : {"logit", "probit", "mlogit", "ologit"}, default="logit"
       Model class used for the data generating process.

   dynamic : {0, 1}, default=0
       If ``dynamic=0``, the regressor matrix are strictly exogenous.
       If ``dynamic=1``, the regressor matrix includes a lagged dependent variable
       (as an additional covariate), producing a specification with predetermined regressors.

   Returns
   -------
   FEi : array_like
       Individual-effect dummy matrix of dimension ``(N·T) × N`` (long-form).

   FEt : array_like
       Time-effect dummy matrix of dimension ``(N·T) × T`` (long-form).

   FE : array_like
       Stacked fixed-effect matrix of dimension ``(N·T) × (N+T)``, constructed by
       concatenating ``FEi`` and ``FEt``.

   index_i : array_like
       Long-form individual indices of dimension ``(N·T) × 1``, taking values in
       ``{1, ..., N}``.

   index_t : array_like
       Long-form time indices of dimension ``(N·T) × 1``, taking values in
       ``{1, ..., T}``.

   Y : array_like
       Outcome variable in long form of dimension ``(N·T) × 1``.

   X : array_like
       Regressors in long form.

       - Static model (``dynamic=0``): ``(N·T) × 1``.
       - Dynamic model (``dynamic=1``): ``(N·T) × 2``, where the first column is the
         lagged dependent variable and the second column is an exogenous/predetermined
         covariate.

   Ystar : array_like
       Latent index in long form (when relevant to the chosen model), dimension ``(N·T) × 1``.

   alphas0 : array_like
       True individual fixed effects (DGP values), typically length ``N``.

   gammas0 : array_like
       True time fixed effects (DGP values), typically length ``T``.

   Notes
   -----
   - The returned objects are in **long form** (dimension proportional to ``N·T``).
     In typical usage with :func:`twowaypanel.fit`, you will reshape the outcome and
     regressors into:

     - ``Y`` as ``(N, T)``
     - ``X`` as ``(N, T, K)``

     where ``K`` is the number of covariates implied by the DGP.

   Examples
   --------
   Generate a dynamic ordered-logit panel and reshape into the format expected by
   :func:`twowaypanel.fit`:

   .. code-block:: python

      FEi, FEt, FE, index_i, index_t, Y, X, Ystar, alphas0, gammas0 = twowaypanel.demo.PanelGenData(
          N=45, T=15, seed=10, model="ologit", dynamic=1
      )

      Y = Y.reshape(45, 15)
      X = X.reshape(45, 15, 2)

      res = twowaypanel.fit(Y, X, model="ologit", prior="Generic", algorithm="JML", cutoff0=-2.5, ape=True)


Empirical data loader
---------------------

``twowaypanel.database.angristevans98``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: twowaypanel.database.angristevans98()

   Load the processed panel dataset based on Angrist and Evans (1998),
   *Children and their parents’ labor supply: Evidence from exogenous variation in family size*
   (*American Economic Review*).

   The dataset is pre-processed and packaged for immediate use in binary logit/probit
   examples. It is intended to demonstrate typical panel-data workflows and serves as
   a convenient benchmark dataset for the package.

   Returns
   -------
   data : object
       A processed dataset object (typically a pandas ``DataFrame``) containing the
       variables needed to construct the panel outcome and regressors used in the
       examples and replication materials.

   Notes
   -----
   - The repository contains the Stata preprocessing scripts and notes used to construct
     the packaged dataset. Please refer to the following folder in the GitHub repository:

     ``twowaypanel/demo/application_angristevans98``

     Source repository:
     `https://github.com/zizhongyan/twowaypanel <https://github.com/zizhongyan/twowaypanel>`__

   - Users typically sort by individual and time indices and reshape the data into
     ``(N, T)`` and ``(N, T, K)`` arrays before calling :func:`twowaypanel.fit`.

   Examples
   --------
   .. code-block:: python

      import twowaypanel

      data = twowaypanel.database.angristevans98()
      # Example workflow (variable names depend on the packaged dataset):
      # data = data.sort_values(["id", "year"])
      # N = data["id"].nunique()
      # T = data["year"].nunique()
      # Y = data["lfp"].to_numpy().reshape(N, T)
      # X = data[["kids0_2", "kids3_5", "kids6_17", "ln_hus_inc"]].to_numpy().reshape(N, T, 4)


Estimation interface
--------------------

``twowaypanel.fit``
~~~~~~~~~~~~~~~~~~

.. py:function:: twowaypanel.fit(Y, X=None, model=None, prior=None, lag=0, ac=False, algorithm="JML", X_names=None, sv=None, silent=False, ape=True, cutoff0=0, mcmc_iters=16000, mcmc_burnin=1000, mcmc_skipsize=2, mcmc_timer=1, mcmc_sv_mle=True, mcmc_diagnosis=True, beta_variance=None, fe_variance=0.3, block_size=8)

   Fit nonlinear panel data models with two-way (individual and time) fixed effects,
   with optional likelihood-based and/or analytical bias correction.

   **Quick pointer.**
   
   - For a user-oriented description of the inputs, options, and outputs, please see
     the *Quick Start Tutorial* section of this documentation.
   - For complete argument descriptions in Python, run ``help(twowaypanel.fit)`` in an
     interactive session.

   Technical notes
   --------------
   - The function expects ``Y`` in ``(N, T)`` form and ``X`` in ``(N, T, K)`` form.
   - Under ``algorithm="JML"``, likelihood-based correction is implemented as a penalty
     term in the objective.
   - Under ``algorithm="MCMC"``, the routine runs Metropolis–Hastings sampling and can
     report convergence diagnostics (e.g., Geweke-type checks and trace/hist/ACF plots).
