# `twowaypanel`: Econometric Analysis of Nonlinear Panel Models with Individual and Time Effects

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/pypi-v.0.9.4-orange)](https://pypi.org/project/twowaypanel/)
[![Docs](https://readthedocs.org/projects/twowaypanel/badge/?version=latest)](https://twowaypanel.readthedocs.io/en/latest/)
[![PyTorch](https://img.shields.io/badge/PyTorch-blue?logo=pytorch&logoColor=white)](https://pytorch.org/)

**twowaypanel** is a Python package for **bias-corrected estimation and inference** in **nonlinear panel data models with additive individual and time fixed effects**.  
It accompanies the paper:

> Yan, Zizhong, Zhengyu Zhang, Mingli Chen, Jingrong Li, Iván Fernández-Val. (2026), *Robust Priors in Nonlinear Panel Models with Individual and Time Effects*. *arXiv e-prints*, arXiv:2604.03663.

**Quick links**
- **Paper**: [arXiv link](https://arxiv.org/abs/2604.03663)
- **Documentation (Read the Docs)**: [twowaypanel.readthedocs.io](https://twowaypanel.readthedocs.io)

## ▶ Overview

Nonlinear panel models with additive individual and time effects are a workhorse in empirical economics. However, fixed-effects maximum likelihood estimation is subject to the **incidental parameter problem**, which can induce substantial **finite-sample bias** in both parameter estimates and economically relevant functionals (e.g., average partial effects).

`twowaypanel` provides **bias-corrected** estimation and inference for:
- model parameters, and  
- **average partial effects (APEs)**, for both **continuous regressors** and **discrete regressors** (finite changes).

The current release supports four model classes—**binary logit**, **binary probit**, **multinomial logit**, and **ordered logit**. The explanatory variables can be either **strictly exogenous** or **predetermined** (e.g., include lagged dependent variables to accommodate dynamic models).


## ▶  Documentation

User-facing documentation (tutorials, examples by model class, and API reference) is maintained on Read the Docs:

- **Documentation homepage**: [twowaypanel.readthedocs.io](https://twowaypanel.readthedocs.io/)
- **Tutorials**: [Quick Start and tutorial pages](https://twowaypanel.readthedocs.io/en/latest/tutorial/tutorial.html)
- **Examples gallery**: [Worked examples by model class](https://twowaypanel.readthedocs.io/en/latest/examples/examples.html)

The documentation explains expected data formats, model options, and how to interpret the printed output tables (including APEs and, when relevant, MCMC diagnostics).


## ▶  Supported models and bias-correction methods

`twowaypanel` implements the likelihood-based bias-correction methods proposed in Yan et al. (2026), including:
- **Integrated-likelihood-based correction** (via *priors*; “prior correction”), and  
- **Joint-likelihood-based correction** (via *penalties*; “penalty correction”).
For logit and probit panels, the package also provides the **analytical bias correction** for fixed-effects MLE developed by Fernández-Val and Weidner (2016).

| Model class       | Regressor exogeneity  | Generic prior / penalty |   Model-specific prior / penalty    | Analytical correction (FW16) |
| ----------------- | :-------------------: | :---------------------: | :---------------------------------: | :--------------------------: |
| Binary logit      |   Strict exogenous    |            ✓            |            ✓ (Prior SE)             |              ✓               |
| Binary logit      |     Predetermined     |            ✓            | ✓ (Prior PE; Prior SE + analytical) |              ✓               |
| Binary Probit     |Strict exog. or predet.|            ✓            |                  –                  |              ✓               |
| Multinomial logit |   Strict exogenous    |            ✓            |            ✓ (Prior SML)            |              –               |
| Multinomial logit |     Predetermined     |            ✓            |            ✓ (Prior PML)            |              –               |
| Ordered logit     |Strict exog. or predet.|            ✓            |                  –                  |              –               |

**Notes.**
- *Model-specific* priors/penalties are defined in Yan et al. (2026) and are designed to deliver improved finite-sample performance within the corresponding model class.
- The *generic* prior/penalty is intended to be **robust across a broad class of nonlinear panel specifications**; see Yan et al. (2026) for details and motivation.



## ▶  Installation 

1. **Installation from PyPI** (recommended)
```bash
pip install twowaypanel
````

2. **Install the development version** from GitHub
```bash
git clone https://github.com/zizhongyan/twowaypanel.git
cd twowaypanel
pip install -e .
```

3. **Install from a local copy**
   
If you downloaded the source code (e.g., as a ZIP file) and unzipped it locally, install from the repository root:
```bash
cd /path/to/twowaypanel
pip install -e .
```

4. **Jupyter / notebook users**
   
To ensure the package is installed into the **same environment as the current Jupyter kernel**, run the following from the repository root directory:
```python
%cd /path/to/twowaypanel
%pip install -e .
```



## ▶ Citing this work

Please cite both the underlying paper and the software package when using this code in your research.

1. **Paper**
   - Yan, Zizhong, Zhengyu Zhang, Mingli Chen, Jingrong Li, Iván Fernández-Val (2026), “Robust priors in nonlinear panel models with individual and time effects.” *arXiv e-prints*, arXiv:2604.03663.

2. **Software**
   - Yan, Zizhong, Zhengyu Zhang, Mingli Chen, Jingrong Li, Iván Fernández-Val (2026),  “twowaypanel: A Python package for econometric analysis of nonlinear panel models with individual and time effects.“  (Version 0.9.4) [Computer software].  Source code: `https://github.com/zizhongyan/twowaypanel`.


## References

- Fernández-Val, Iván and Martin Weidner (2016),  “Individual and time effects in nonlinear panel models with large N, T.” *Journal of Econometrics*, 192(1), 291–312.

---

**Code maintainer:** Zizhong Yan, Institute for Economic and Social Research (IESR), Jinan University, Guangzhou, China.  
Email: `helloyzz@gmail.com`



