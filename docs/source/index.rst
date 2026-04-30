twowaypanel 
============

Welcome to the documentation for **twowaypanel**, a Python package for **bias-corrected estimation and inference**
in **nonlinear panel data models with two-way (additive individual and time) fixed effects**.
This package accompanies the paper:

    Yan, Zizhong, Zhengyu Zhang, Mingli Chen, Jingrong Li, Iván Fernández-Val. (2026),
    *Robust Priors in Nonlinear Panel Models with Individual and Time Effects*.
    *arXiv e-prints*, arXiv:2604.03663.

For the source code, release notes, and issue tracking, please visit the project’s GitHub homepage:
`https://github.com/zizhongyan/twowaypanel <https://github.com/zizhongyan/twowaypanel>`_

`twowaypanel` provides **bias-corrected** estimation and inference for:

- model parameters, and  
- **average partial effects (APEs)**, for both **continuous regressors** and **discrete regressors** (finite changes).

In the current release, `twowaypanel` supports four model classes: **binary logit**, **probit**, **multinomial logit**, and **ordered logit**. The explanatory variables can be either **strictly exogenous** or **predetermined** (e.g., include lagged dependent variables to accommodate dynamic models).

If you are new to the package, we recommend starting with **Installation**, then working through the **Tutorial**
and **Examples** sections, and finally consulting the **API Reference** for detailed function/class documentation.


Table of Contents
=================

.. toctree::
   :maxdepth: 3

   install/install

.. toctree::
   :maxdepth: 3

   tutorial/tutorial

.. toctree::
   :maxdepth: 2

   examples/examples

.. toctree::
   :maxdepth: 2

   api/api



Code maintainer
---------------

**Zizhong Yan**, Institute for Economic and Social Research (IESR), Jinan University, Guangzhou, China.  
Email: `helloyzz@gmail.com <mailto:helloyzz@gmail.com>`_