"""
Version: v.0.9.4-beta

Created on Tue Aug 26 2025

Last modified on Mon Oct 27 2025

Author: Zizhong Yan @ copyright
"""
#----------------------------------------------------------
# Load library dependencies
#----------------------------------------------------------
import sys
import os
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.panelMLE import panelFits, cutoff_start_paramsfreq, cutoffTrans, cutoffTransReverse
#----------------------------------------------------------
# Class of nonlinear panel data regression
#----------------------------------------------------------
class fit:
    """
    Fit nonlinear panel data models with two-way (individual and time) fixed effects,
    with optional likelihood-based and/or analytical bias correction.

    This is the main entry point of **twowaypanel**. It implements likelihood-based
    bias corrections proposed in Yan et al. (2026), including:

    - Integrated-likelihood-based correction (via *priors*; "prior correction")
    - Joint-likelihood-based correction (via *penalties*; "penalty correction")

    For binary logit and probit panels, it also provides the analytical bias correction
    for fixed-effects MLE developed by Fernández-Val and Weidner (2016).

    Parameters
    ----------
    Y : array_like
        Dependent variable arranged as an ``(N, T)`` array (typically a 2D NumPy array),
        where ``N`` is the number of individuals and ``T`` is the number of time periods.
        This argument is required.

    X : array_like, 
        Regressors arranged as an ``(N, T, K)`` array (3D NumPy array), where ``K`` is the
        number of covariates. This argument is required in typical use.

        Convention:
        - first index: individual i = 1,...,N
        - second index: time t = 1,...,T
        - third index: covariate k = 1,...,K

    model : {"logit", "probit", "ologit", "mlogit"}, required
        Model class (likelihood) to be estimated:

        - ``"logit"``  : binary logit
        - ``"probit"`` : binary probit
        - ``"ologit"`` : ordered logit
        - ``"mlogit"`` : multinomial logit

    prior : {None, "Generic", "SE", "PE", "SML", "PML"}, optional
        Selects the likelihood-based correction. If ``None`` (default), no likelihood-based
        correction is applied.

        Common choices (following Yan et al., 2026):
        - ``"Generic"`` : robust correction for a broad class of nonlinear panels
        - ``"SE"``      : for binary logit with strictly exogenous regressors
        - ``"PE"``      : for binary logit with predetermined regressors
        - ``"SML"``     : for multinomial logit with strictly exogenous regressors
        - ``"PML"``     : for multinomial logit with predetermined regressors

        Note: a single keyword ``prior`` is used to control both "prior correction" and
        "penalty correction"; the interpretation depends on ``algorithm``.

    lag : int, default=0
        If ``lag=0``, all regressors are treated as strictly exogenous.
        In settings with predetermined regressors, the trimming parameter ``lag`` controls 
        the truncation level for estimating spectral expectation objects entering bias correction. 
        The same ``lag`` is used when trimming is required by analytical correction.

        Larger ``lag`` allows richer serial dependence to enter the approximation, at the
        cost of more computation.

    ac : bool, default=False
        If ``True``, apply analytical bias correction on the estimator (FW16). This option
        is effective for binary logit/probit models.

        - If ``ac=True`` and ``prior is None``, applies Fernández-Val and Weidner (2016).
        - In the logit with predetermined regressors, if ``prior="SE"`` and ``ac=True``, workflow
          follows Yan et al. (2026): partially correct the likelihood via the prior, then
          remove remaining cross-time dependence bias via an additional analytical step.

    algorithm : {"JML", "MCMC"}, default="JML"
        Estimation algorithm:

        - ``"JML"``  : joint maximum likelihood estimation. In this case, the chosen
          likelihood-based correction is implemented as a *penalty* added to the likelihood.
        - ``"MCMC"`` : Metropolis–Hastings sampling for the prior-based formulation.

    X_names : list of str, optional
        Variable labels for readable output. If provided, must be a list of length ``K``.
        If omitted, default labels ``X_1, X_2, ...`` are used.

    sv : object, optional
        Starting values used by the optimizer (and/or to initialize MCMC, depending on
        settings). In most applications, defaults are sufficient.

    silent : bool, default=False
        If ``True``, suppress printing of the summary table. Results are still returned.

    ape : bool, default=True
        If ``True``, compute average partial effects (APEs) in addition to parameter
        estimates. Both continuous-regressor APEs (marginal effects) and discrete-regressor
        APEs (finite changes) are supported.

    cutoff0 : float, default=0
        Ordered-logit identification normalization: the first cutoff is fixed at ``cutoff0``.
        Only used when ``model="ologit"``.

    mcmc_iters : int, default=16000
        Number of MCMC iterations (only when ``algorithm="MCMC"``).

    mcmc_burnin : int, default=1000
        Burn-in iterations (only when ``algorithm="MCMC"``).

    mcmc_skipsize : int, default=2
        Thinning: keep one draw every ``mcmc_skipsize`` iterations (only when ``algorithm="MCMC"``).

    mcmc_timer : int, default=1
        Progress reporting frequency for MCMC (only when ``algorithm="MCMC"``).

    mcmc_sv_mle : bool, default=True
        If ``True``, initialize MCMC at MLE-based starting values (often helpful). Only used
        when ``algorithm="MCMC"``.

    mcmc_diagnosis : bool, default=True
        If ``True``, produce basic MCMC diagnostics (e.g., Geweke tests) and common plots such
        as trace plots, histograms, and autocorrelation functions (ACFs) for selected common
        parameters. Only used when ``algorithm="MCMC"``.

    beta_variance : float or array_like, optional
        Proposal variance for the Gaussian random-walk MH updates of common parameters during
        early iterations (first ~400 iterations). Afterwards, a self-adaptive scheme is used.
        Only used when ``algorithm="MCMC"``.

    fe_variance : float, default=0.3
        Proposal variance for the Gaussian random-walk MH updates of fixed effects during
        early iterations (first ~400 iterations). Afterwards, a self-adaptive scheme is used.
        Only used when ``algorithm="MCMC"``.

    block_size : int, default=8
        Block size for blockwise MH updates (only when ``algorithm="MCMC"``).

    Returns
    -------
    result : object
        A result object storing estimates, standard errors, and (optionally) APEs.

        Main attributes (available under both ``algorithm="JML"`` and ``algorithm="MCMC"``):
        - ``paras``   : estimated common parameters (e.g., regression coefficients)
        - ``se``      : asymptotic standard errors for ``paras``
        - ``feparas`` : estimated fixed effects parameters (individual and time effects)
        - ``success`` : success indicator (1 if estimation completed successfully)
        - ``fun``     : final value of the objective function (e.g., final (penalized) log-likelihood)
        - ``ape``     : APE estimates (when ``ape=True``)
        - ``apese``   : asymptotic standard errors for APEs (when ``ape=True``)

        Additional attributes when ``algorithm="MCMC"``:
        - ``theta_mcmc``    : MCMC draws for common parameters
        - ``lambda_mcmc``   : MCMC draws for fixed effects parameters
        - ``sd``            : Monte Carlo standard deviations of retained draws
        - ``accRateTheta``  : acceptance rates for common-parameter updates
        - ``accRateFE``     : acceptance rates for fixed-effects updates

    Notes
    -----
    - For models with predetermined regressors, the trimming parameter ``lag`` is used for
      estiamting spectral-expectation objects entering bias-correction terms.

    Examples
    --------
    >>> import twowaypanel
    >>> # Y: (N, T), X: (N, T, K)
    >>> res = twowaypanel.fit(Y, X, model="logit", prior="SE", algorithm="JML", ape=True)
    >>> res.paras
    >>> res.se
    >>> res.ape
    """
    def __init__(self, Y, X=None, model=None,
                  prior=None, lag=0, ac=False,
                  drop_separation=True, algorithm="JML",
                  X_names=None, sv=None, silent=False,ape=True,cutoff0=0,
                  mcmc_iters=16000,mcmc_burnin=1000,mcmc_skipsize=2,mcmc_timer=1,mcmc_sv_mle=True,
                  mcmc_save=False,mcmc_savefolder=None,mcmc_savesuffix=None,mcmc_diagnosis=True,
                  beta_variance=None,fe_variance=0.3,block_size=8):
        #----------------------------------------------------------
        # Preparations
        #----------------------------------------------------------
        # [> Check compatibility of arguments <]
        if model !="logit" and model != "probit" and model != "ologit" and model != "mlogit":
            sys.exit("Error: `model` option is not correctly defined. Please see the helpfile: help(fit)")
        if prior is not None and prior != "Binary" and prior != "Generic" and prior != "StaticExp" and prior != "SE" and prior != "PE" and prior != "SML" and prior != "PML":
            sys.exit("Error: `prior` option is not correctly defined. Please see the helpfile: help(fit)")
        if algorithm != "JML" and algorithm != "Iter" and algorithm != "MCMC":
            sys.exit("Error: `algorithm` option is not correctly defined. Please see the helpfile: help(fit)")
        # [> Check compatibility of input variables <]
        if Y.ndim!=2 or np.shape(Y)[1]==1: 
            sys.exit("Error: dependent variable Y is not an N by T 2d NumPy array. Please see the helpfile: help(fit)")
        if np.shape(Y)[1]==1: 
            sys.exit("Error: dependent variable Y is not correctly defined. Please see the helpfile: help(fit)")
        if model=="probit" or model=="logit":
            if np.all(np.unique(Y)==np.array([0,1]))!=True:
                sys.exit("Error: dependent variable Y is not correctly defined, or is not binary.")
        ebc=ac
        if ebc is False and prior is None: lag=0
        if model == "mlogit": ebc = False 
        if prior == "SE": prior = "StaticExp"
        if prior == "SML": prior = "StaticExp"
        if prior == "PE": prior = "Binary"
        if prior == "PML": prior = "Binary"
        # Check covariates
        if X.ndim!=3: sys.exit("Error: covariate X is not an N by T by K 3d NumPy array. Please see the helpfile: help(fit)")
        # [> Drop the separated data in the binary data case -- if option is specified <]
        self.N = np.shape(Y)[0]
        self.T = np.shape(Y)[1]
        self.Nold = np.copy(self.N)
        self.Told = np.copy(self.T)
        self.separated=0 
        if model != "mlogit":
            if drop_separation==True and model != "ologit":
                if np.unique(Y.sum(axis=0))[0]==0 or np.unique(Y.sum(axis=1))[0]==0:
                    X = X[Y.sum(axis=1)!=0,:,:]
                    Y = Y[Y.sum(axis=1)!=0,:]
                    X = X[:,Y.sum(axis=0)!=0,:]
                    Y = Y[:,Y.sum(axis=0)!=0]
                    self.N = np.shape(Y)[0]
                    self.T = np.shape(Y)[1]
                    self.separated=1
                if np.unique(Y.sum(axis=0))[-1]==self.N or np.unique(Y.sum(axis=1))[-1]==self.T:
                    X = X[Y.sum(axis=1)!=self.T,:,:]
                    Y = Y[Y.sum(axis=1)!=self.T,:]
                    X = X[:,Y.sum(axis=0)!=self.N,:]
                    Y = Y[:,Y.sum(axis=0)!=self.N]
                    self.N = np.shape(Y)[0]
                    self.T = np.shape(Y)[1]
                    self.separated=1
        if model == "mlogit" and drop_separation==True:
            NN = np.shape(Y)[0]
            TT = np.shape(Y)[1]
            JJ=np.size(np.unique(Y))
            for reps in range(max(NN,TT)):
                self.N = np.shape(Y)[0]
                self.T = np.shape(Y)[1]
                iMask = np.zeros(self.N)
                tMask = np.zeros(self.T)
                for ii in range(self.N): 
                    if np.size(np.unique(Y[ii,:]))<JJ:
                        iMask[ii] = 1
                X = X[iMask==0,:,:]
                Y = Y[iMask==0,:]
                for tt in range(self.T): 
                    if np.size(np.unique(Y[:,tt]))<JJ:
                        tMask[tt] = 1
                X = X[:,tMask==0,:]
                Y = Y[:,tMask==0]

                self.N = np.shape(Y)[0]
                self.T = np.shape(Y)[1]
            if NN!=self.N or TT!=self.T: self.separated=1
        # [> For ordinal and multinomial response model, normalize Y's value starting from zero <]
        if model == "ologit" and np.min(Y)!=0:
            Y = Y-np.min(Y)
        if model == "mlogit" and np.min(Y)!=0:
            Y = Y-np.min(Y)
        # [> Organize variables <]
        # Dependent variable and covariates
        N = np.shape(Y)[0]
        T = np.shape(Y)[1]
        dimX = np.shape(X)[2]
        Y=Y.reshape(-1,1)
        X=X.reshape(-1,dimX)
        # Create fixed effects dummies regressors
        FEi = np.repeat(np.eye(N),T,axis=0)[:,1:]
        FEt = np.tile(np.eye(T),N).T
        FE = np.hstack((FEi,FEt))
        # [> Starting values of parameters in the optimization <]
        # If not provided, starting values are all zeroes by default
        if model == "logit" or model == "probit":
            initials = np.zeros(N+T-1+dimX)
        if model == "ologit":
            # Cutoff parameters are based on the bins of Y and transformed for the identification purpose.
            initials=np.zeros(N+T-1+dimX+int(np.max(Y))-1)
            initials[-(int(np.max(Y))-1):]= cutoff_start_paramsfreq(Y)[1:]
            # Starting values of FE parameters is randomly initialized by non-zero small values
            np.random.seed(0)
            initials[:N+T-1] = np.append(stats.norm.rvs(size=N-1),stats.norm.rvs(size=T))
        if model == "mlogit":
            Ymax = int(np.max(Y))
            initials = np.zeros((np.shape(np.hstack((FE,X)))[1],Ymax)).reshape(-1)
        # With provided sv (starting values)
        if sv is not None: 
            if np.size(sv)!=np.size(initials):
                sys.exit("Error: the number of starting values does not match the number of parameters.")
            else:
                initials=sv.reshape(-1)
        else:
            sv = np.copy(initials)
        # [> Change all input variables to float 64bit <]
        if Y.dtype  != 'float64': Y  = Y.astype('float64')
        if X.dtype  != 'float64': X  = X.astype('float64')
        if sv.dtype != 'float64': sv = sv.astype('float64')
        # [> Setups for MCMC <]
        # `fe_variance` and `beta_variance` set the variances in the random walk proposal for initial 400 runs, 
        # rest runs are based on self-adapted proposals
        # `block_size` sets the size of random blocked parameters
        if beta_variance is None: 
            if model != "mlogit":
                beta_variance = np.sqrt(0.1)*np.ones(dimX)
            if model == "mlogit":
                beta_variance = np.sqrt(0.1)*np.ones(dimX*Ymax)
        if model == "logit":
            para_variance = np.concatenate((fe_variance*np.ones(self.N+self.T-1),beta_variance))
        if model == "probit":
            para_variance = np.concatenate((fe_variance*np.ones(self.N+self.T-1),beta_variance))
        if model == "ologit":
            cutoff_variance = np.sqrt(0.01)
            para_variance = np.concatenate((fe_variance*np.ones(self.N+self.T-1),beta_variance,cutoff_variance*np.ones(int(np.max(Y))-1)))
        if model == "mlogit":
            para_variance = np.concatenate((fe_variance*np.ones(int((self.N+self.T-1)*Ymax)),beta_variance))
        # [> Check whether there are dummy regressors -- mark for generating correct APE <]
        dummyIndicator = np.zeros(dimX,dtype='int')
        if ape==True:
            for kx in range(dimX):
                if np.unique(X[:,kx]).size==2:
                    dummyIndicator[kx] = np.array_equal(np.unique(X[:,kx]), np.array([0,1]))
        #----------------------------------------------------------
        # Estimation
        #----------------------------------------------------------
        self.success = 1
        try:
            (self.paras, self.se, self.feparas, 
                self.success, self.fun, end_time, 
                self.ape, self.apese,
                self.theta_mcmc,self.lambda_mcmc,
                self.sd,
                self.accRateTheta ,self.accRateFE) = panelFits(Y=Y, X=X, FEi=FEi,FEt=FEt,FE=FE, N=self.N, T=self.T, 
                                                                Nold=self.Nold,Told=self.Told, 
                                                                dimX=dimX,dummyIndicator=dummyIndicator,
                                                                model=model, priorVersion=prior,  priorLag=lag, algorithm=algorithm, ebc=ebc,
                                                                sv=sv, silent=silent,ape_compute=ape,seps=self.separated,cutoff0=cutoff0,
                                                                mcmc_iters=mcmc_iters,mcmc_burnin=mcmc_burnin,mcmc_skipsize=mcmc_skipsize,
                                                                mcmc_timer=mcmc_timer,mcmc_sv_mle=mcmc_sv_mle,
                                                                para_sd=np.sqrt(para_variance),block_size=block_size)
        except:
            self.success = 0
            sys.exit("Estimation failed.")
        #----------------------------------------------------------
        # Save the chain of results to local HD, if specified
        #----------------------------------------------------------
        if mcmc_save == True and algorithm=="MCMC":
            if not os.path.exists(mcmc_savefolder): os.makedirs(mcmc_savefolder)
            np.savetxt(mcmc_savefolder+"/resultPara"+mcmc_savesuffix+".csv",
                       np.hstack((self.theta_mcmc,self.lambda_mcmc)), fmt='%6.5f')
        #----------------------------------------------------------
        # Broadcasting
        #----------------------------------------------------------
        # Functions for Geweke (1992) test
        def splitSample(a, n):
            k, m = divmod(len(a), n)
            return list((a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)))
        def geweke(chain):
            sampleSize= np.shape(chain)[0]    
            firstSample = chain[:int(sampleSize/10)]
            lastSampleList = splitSample(chain[int(sampleSize/50):], 20)
            statistic = np.zeros(20)
            for ii in range(20):
                statistic[ii] =  (firstSample.mean() - lastSampleList[ii].mean()) /np.sqrt(firstSample.var() + lastSampleList[ii].var())
            pval = 2*stats.norm.sf(np.abs(statistic))
            return statistic,pval,np.mean(pval),np.min(pval)
        if silent is False: 
            print("",)
            print("--------------------------------------------------------------------------------")            
            #print('-' * 80)
            print("---- ESTIMATION RESULTS --------------------------------------------------------")             
            if model=='logit':
                print("                 LOGIT PANEL MODEL WITH TWO-WAY FIXED EFFECTS")
            if model=='probit':
                print("                 PROBIT PANEL MODEL WITH TWO-WAY FIXED EFFECTS")
            if model=='ologit':
                print("               ORDERED LOGIT PANEL MODEL WITH TWO-WAY FIXED EFFECTS")
            if model=='mlogit':
                print("             MULTINOMIAL LOGIT PANEL MODEL WITH TWO-WAY FIXED EFFECTS")
            if prior=='Generic':
                print("                  USING BIAS-REDUCING PRIOR (GENERIC VERSION)")
            if prior=='StaticExp':
                print("             USING BIAS-REDUCING PRIOR FOR EXPONENTIAL FAMILY MODELS")
                print("                      WITH STRICTLY EXOGENOUS REGRESSORS")
            if prior==None and ebc==True:
                print("                ANALYTICAL BIAS CORRECTION ON ESTIMATOR (FW16)")
            if prior is not None and ebc==True:
                print("                FURTHER ANALYTICAL BIAS CORRECTION ON ESTIMATOR")
            if prior=='Binary':
                print("             USING BIAS-REDUCING PRIOR FOR EXPONENTIAL FAMILY MODELS")
                print("                         WITH PREDETERMINED REGRESSORS")
            if prior==None and ebc is not True:
                print("                           WITHOUT BIAS CORRECTION")
            print("--------------------------------------------------------------------------------")            
            print("Number of Individuals:  %4s                              Observations: %8s" % (self.N,self.N*self.T))
            if algorithm!="MCMC":
                print("Number of Time Periods: %4s                            Log-likelihood: %8.2f" % (self.T,self.fun))
            if algorithm=="MCMC":
                print("Number of Time Periods: %4s   " % (self.T))
            if algorithm=="JML":
                print("Algorithm: Joint MLE                              Time spent (seconds): %8.3f" % end_time)
            if algorithm=="MCMC":
                print("Algorithm: Markov Chain Monte Carlo               Time spent (minutes): %8.3f" % (end_time/60))
                print("           %s reptitions" % mcmc_iters)
            if algorithm=="Iter":
                print("Algorithm: Iterative MLE (concentration scheme)   Time spent (seconds): %8.3f" % end_time)
            print("--------------------------------------------------------------------------------")            
            if model!='ologit': 
                print("Independent variable    Coefficient     Std. Err.   P>|z|   [95% conf. interval]")
            if model=='ologit': 
                print("                        Coefficient     Std. Err.   P>|z|   [95% conf. interval]")
            print("--------------------------------------------------------------------------------")  
            if X_names is None:
                X_names = []
                for kk in range(0,dimX):
                    X_names.append("X" + str(kk+1))
            if model=='ologit': 
                C_names = []
                for kk in range(dimX,dimX+int(np.max(Y))):
                    C_names.append("/cutoff" + str(kk-dimX+2))
            if model!='mlogit':
                for kk in range(0,dimX):
                    print("%20s%15s%14s%8.3f%12s%11s" % (X_names[kk][:15],
                                                str(self.paras[kk])[:11],
                                                str(self.se[kk])[:10],
                                                2*sp.stats.norm.sf(abs(self.paras[kk]/self.se[kk])),
                                                str(self.paras[kk]-1.9599*self.se[kk])[:8]  ,
                                                str(self.paras[kk]+1.9599*self.se[kk])[:8]  ))
            if model=='mlogit':
                for jj in range(0,Ymax):
                    print("  Y =", jj+2)  
                    for kk in range(0,dimX):
                        print("%20s%15s%14s%8.3f%12s%11s" % (X_names[kk][:15],
                                                    str(self.paras[kk,jj])[:11],
                                                    str(self.se[kk,jj])[:10],
                                                    2*sp.stats.norm.sf(abs(self.paras[kk,jj]/self.se[kk,jj])),
                                                    str(self.paras[kk,jj]-1.9599*self.se[kk,jj])[:8]  ,
                                                    str(self.paras[kk,jj]+1.9599*self.se[kk,jj])[:8]  ))

            if model=='ologit': 
                print("--------------------------------------------------------------------------------")            
                for kk in range(dimX,dimX+int(np.max(Y))-1):
                    print("%20s%15s%14s%8.3f%12s%11s" % (C_names[kk-dimX][:15],
                                                str(self.paras[kk])[:11],
                                                str(self.se[kk])[:10],
                                                2*sp.stats.norm.sf(abs(self.paras[kk]/self.se[kk])),
                                                str(self.paras[kk]-1.9599*self.se[kk])[:8]  ,
                                                str(self.paras[kk]+1.9599*self.se[kk])[:8]  ))
            
            print("--------------------------------------------------------------------------------")     

            if ape==True:
                print("")
                print("--------------------------------------------------------------------------------")     
                if model!='ologit': 
                    apedisp = self.ape
                    print("---- AVERAGE PARTIAL EFFECTS ---------------------------------------------------")
                    print("--------------------------------------------------------------------------------")            
                    print("Independent variable    Coefficient     Std. Err.   P>|z|   [95% conf. interval]")
                    print("--------------------------------------------------------------------------------")            
                    if model!='mlogit': 
                        for kk in range(0,dimX):
                            print("%20s%15s%14s%8.3f%12s%11s" % (X_names[kk][:15],
                                                        str(apedisp[kk])[:11],
                                                        str(self.apese[kk])[:10],
                                                        2*sp.stats.norm.sf(abs(apedisp[kk]/self.apese[kk])),
                                                        str(apedisp[kk]-1.9599*self.apese[kk])[:8]  ,
                                                        str(apedisp[kk]+1.9599*self.apese[kk])[:8]  ))             
                    if model=='mlogit': 
                        for jj in range(0,Ymax):
                            print("  Y =", jj+2)  
                            for kk in range(0,dimX):
                                print("%20s%15s%14s%8.3f%12s%11s" % (X_names[kk][:15],
                                                            str(apedisp[kk,jj])[:11],
                                                            str(self.apese[kk,jj])[:10],
                                                            2*sp.stats.norm.sf(abs(apedisp[kk,jj]/self.apese[kk,jj])),
                                                            str(apedisp[kk,jj]-1.9599*self.apese[kk,jj])[:8]  ,
                                                            str(apedisp[kk,jj]+1.9599*self.apese[kk,jj])[:8]  ))      
                if model=='ologit': 
                    print("---- AVERAGE PARTIAL EFFECTS ---------------------------------------------------")
                    for ylevel in range(int(np.max(Y))+1):
                        apedisp = self.ape[:,ylevel]
                        apeSEdisp = self.apese[:,ylevel]
                        ylevelp1 = ylevel+1
                        print("--------------------------------------------------------------------------------")            
                        print("             Pr(Y=%1.0f)    Coefficient     Std. Err.   P>|z|   [95%% conf. interval]" % ylevelp1)
                        print("--------------------------------------------------------------------------------")            
                        for kk in range(0,dimX):
                            print("%20s%15s%14s%8.3f%12s%11s" % (X_names[kk][:15],
                                                        str(apedisp[kk])[:11],
                                                        str(apeSEdisp[kk])[:10],
                                                        2*sp.stats.norm.sf(abs(apedisp[kk]/apeSEdisp[kk])),
                                                        str(apedisp[kk]-1.9599*apeSEdisp[kk])[:8]  ,
                                                        str(apedisp[kk]+1.9599*apeSEdisp[kk])[:8]  ))
                print("--------------------------------------------------------------------------------")  
            if lag!=0:
                print("Note: The trimming parameter for estimating spectral expectation is %1s." % lag)
            if self.separated==1:
                print("Note: Panel data contains under-identified observations;")
                print("      Dropped %1.0f out of %1.0f individuals, and %1.0f out of %1.0f time periods. " % (self.Nold-self.N,self.Nold,self.Told-self.T,self.T))


            if algorithm=="MCMC" and mcmc_diagnosis==True:
                print("")
                print("--------------------------------------------------------------------------------")
                print("---- MCMC DIAGNOSIS ------------------------------------------------------------")
                print("--------------------------------------------------------------------------------")
                print("                                       Geweke (1992) Convergence Diagnostic Test")
                print("                                             First 10% sample vs. 20 segments ")
                print("                                                   of final 50% sample")
                print("Independent variable    Acceptance(%)    Average p-value       Smallest p-value ")
                print("--------------------------------------------------------------------------------")
                if model!='mlogit':
                    for kk in range(0,dimX):
                        _,_,avep,minp = geweke(self.theta_mcmc[:,kk])
                        print("%20s%13s%%%17.3f%22.3f" % (X_names[kk][:15],
                                                    str(self.accRateTheta[kk])[:9], avep,minp))

                if model=='mlogit':
                    for jj in range(0,Ymax):
                        print("  Y =", jj+1)  
                        for kk in range(0,dimX):
                            _,_,avep,minp = geweke(self.theta_mcmc[:,kk,jj])
                            print("%20s%13s%%%17.3f%22.3f" % (X_names[kk][:15],
                                                        str(self.accRateTheta[kk,jj])[:9], avep,minp))

                print("--------------------------------------------------------------------------------")     
                if model=='mlogit':
                    if dimX==1: sLk=6; varNAMES=["$\\beta_{11}$","$\\beta_{12}$","$\\alpha_{21}$","$\\alpha_{N2}$","$\\gamma_{11}$","$\\gamma_{T2}$"]
                    if dimX>=2: sLk=8; varNAMES=["$\\beta_{11}$","$\\beta_{21}$","$\\beta_{12}$","$\\beta_{22}$","$\\alpha_{21}$","$\\alpha_{N2}$","$\\gamma_{11}$","$\\gamma_{T2}$"]
                    for vv in range(sLk-4):
                        if dimX==1:
                            if vv==0: chain = self.theta_mcmc[:,0,0].reshape(-1,1)
                            if vv==1: chain = self.theta_mcmc[:,0,1].reshape(-1,1)
                            if vv==2: chain = self.lambda_mcmc[:,0       ,0].reshape(-1,1)
                            if vv==3: chain = self.lambda_mcmc[:,self.N-1,1].reshape(-1,1)
                            if vv==4: chain = self.lambda_mcmc[:,self.N  ,0].reshape(-1,1)
                            if vv==5: chain = self.lambda_mcmc[:,-1,     -1].reshape(-1,1)
                        if dimX>=2:
                            if vv==0: chain = self.theta_mcmc[:,0,0].reshape(-1,1)
                            if vv==1: chain = self.theta_mcmc[:,1,0].reshape(-1,1)
                            if vv==2: chain = self.theta_mcmc[:,0,1].reshape(-1,1)
                            if vv==3: chain = self.theta_mcmc[:,1,1].reshape(-1,1)
                            if vv==4: chain = self.lambda_mcmc[:,0       ,0].reshape(-1,1)
                            if vv==5: chain = self.lambda_mcmc[:,self.N-1,1].reshape(-1,1)
                            if vv==6: chain = self.lambda_mcmc[:,self.N  ,0].reshape(-1,1)
                            if vv==7: chain = self.lambda_mcmc[:,-1,     -1].reshape(-1,1)
                        print("")
                        df_chain = pd.DataFrame(chain,columns=['theta'])
                        plt.figure(figsize=(8,5))
                        plt.subplot(211)
                        plt.plot(np.arange(chain.shape[0]),chain,label="MCMC sample",color="grey",lw=0.8)
                        plt.plot(np.arange(chain.shape[0]), df_chain.theta.rolling(window=200,center=False).mean(),label="Moving average",color="k",lw=1.5)
                        plt.legend(fontsize=10,loc='lower right', ncol=2)
                        plt.title('Trace Plot: '+varNAMES[vv], fontsize=12)

                        plt.subplot(223)
                        plt.hist(chain,bins=15, color='white', edgecolor='grey',density=True)
                        sns.kdeplot(chain.reshape(-1), color='k', fill=False, bw_method=1/4,linewidth=1.2)
                        plt.title('Histogram: '+varNAMES[vv], fontsize=12)

                        plt.subplot(224)
                        markers, stemline, baseline, = plt.stem(np.arange(100),sm.tsa.acf(chain, nlags = len(range(100))-1),linefmt="grey",markerfmt="Dk")
                        plt.setp(markers, markersize=1.2, markeredgecolor="k",  linewidth=0.2)
                        plt.setp(baseline, 'color', 'k', 'linewidth', 1)
                        plt.title('Auto correlation function: '+varNAMES[vv], fontsize=12)

                        plt.tight_layout()
                        plt.show() 
                if model!='mlogit':
                    if dimX==1: sLk=5; varNAMES=["$\\beta_1$","$\\alpha_2$","$\\alpha_{N}$","$\\gamma_{1}$","$\\gamma_{T}$"]
                    if dimX>=2: sLk=6; varNAMES=["$\\beta_1$","$\\beta_2$","$\\alpha_2$","$\\alpha_{N}$","$\\gamma_{1}$","$\\gamma_{T}$"]
                    if dimX>=4: sLk=8; varNAMES=["$\\beta_1$","$\\beta_2$","$\\beta_3$","$\\beta_4$","$\\alpha_2$","$\\alpha_{N}$","$\\gamma_{1}$","$\\gamma_{T}$"]
                    for vv in range(sLk-4):
                        if dimX==1:
                            if vv==0: chain = self.theta_mcmc[:,0].reshape(-1,1)
                            if vv==1: chain = self.lambda_mcmc[:,0].reshape(-1,1)
                            if vv==2: chain = self.lambda_mcmc[:,self.N].reshape(-1,1)
                            if vv==3: chain = self.lambda_mcmc[:,self.N+1].reshape(-1,1)
                            if vv==4: chain = self.lambda_mcmc[:,-1].reshape(-1,1)
                        if dimX>=2:
                            if vv==0: chain = self.theta_mcmc[:,0].reshape(-1,1)
                            if vv==1: chain = self.theta_mcmc[:,1].reshape(-1,1)
                            if vv==2: chain = self.lambda_mcmc[:,0].reshape(-1,1)
                            if vv==3: chain = self.lambda_mcmc[:,self.N].reshape(-1,1)
                            if vv==4: chain = self.lambda_mcmc[:,self.N+1].reshape(-1,1)
                            if vv==5: chain = self.lambda_mcmc[:,-1].reshape(-1,1)
                        if dimX>=4:
                            if vv==0: chain = self.theta_mcmc[:,0].reshape(-1,1)
                            if vv==1: chain = self.theta_mcmc[:,1].reshape(-1,1)
                            if vv==2: chain = self.theta_mcmc[:,2].reshape(-1,1)
                            if vv==3: chain = self.theta_mcmc[:,3].reshape(-1,1)
                            if vv==4: chain = self.lambda_mcmc[:,0].reshape(-1,1)
                            if vv==5: chain = self.lambda_mcmc[:,self.N].reshape(-1,1)
                            if vv==6: chain = self.lambda_mcmc[:,self.N+1].reshape(-1,1)
                            if vv==7: chain = self.lambda_mcmc[:,-1].reshape(-1,1)
                        print("")
                        df_chain = pd.DataFrame(chain,columns=['theta'])
                        plt.figure(figsize=(8,5))
                        plt.subplot(211)
                        plt.plot(np.arange(chain.shape[0]),chain,label="MCMC sample",color="grey",lw=0.8)
                        plt.plot(np.arange(chain.shape[0]), df_chain.theta.rolling(window=200,center=False).mean(),label="Moving average",color="k",lw=1.5)
                        plt.legend(fontsize=10,loc='lower right', ncol=2)
                        plt.title('Trace Plot: '+varNAMES[vv], fontsize=12)

                        plt.subplot(223)
                        plt.hist(chain,bins=15, color='white', edgecolor='grey',density=True)
                        sns.kdeplot(chain.reshape(-1), color='k', fill=False, bw_method=1/4,linewidth=1.2)
                        plt.title('Histogram: '+varNAMES[vv], fontsize=12)

                        plt.subplot(224)
                        markers, stemline, baseline, = plt.stem(np.arange(100),sm.tsa.acf(chain, nlags = len(range(100))-1),linefmt="grey",markerfmt="Dk")
                        plt.setp(markers, markersize=1.2, markeredgecolor="k",  linewidth=0.2)
                        plt.setp(baseline, 'color', 'k', 'linewidth', 1)
                        plt.title('Auto correlation function: '+varNAMES[vv], fontsize=12)

                        plt.tight_layout()
                        plt.show()
            print("")
#not oOo