"""
Created on Tue Aug 26 2025

Author: Zizhong Yan
"""
#----------------------------------------------------------
# Load library dependencies
#----------------------------------------------------------
import time
import os
import numpy as np
import scipy as sp
from pandas import get_dummies
from scipy import optimize,stats,linalg
from tqdm import tqdm
import torch
from ..lib.torchmin import minimize as minimize2
#----------------------------------------------------------
# Functions for Cutoff points in Ordered Logit Setting
#----------------------------------------------------------
# Starting cutoff parameters for the optimizing the ordinal response model. First cutoff point is fixed.
def cutoff_start_paramsfreq(Y):
    freq = np.bincount(Y.astype('int').reshape(-1))/len(Y.astype('int').reshape(-1))
    params = stats.logistic.ppf(np.clip(freq.cumsum(), 0, 1))
    return np.concatenate((params[:1],np.log(np.diff(params[:-1]))))
# Transformation of the cutoff parameters in the optimizing the ordinal response model
def cutoffTrans(arr,anchor):
    '''
    The first, lowest threshold is unchanged, 
    all other thresholds are in terms of exponentiated increments.
    
    anchor: value of the first fixed cutoff point 
    arr: values for remaining cutoffs
    '''
    return np.append(anchor,np.exp(arr.reshape(-1))).cumsum()[1:]
# Reverse transformation of the cutoff parameters in the optimizing the ordinal response model
def cutoffTransReverse(arr,anchor):
    '''
    anchor: value of the first fixed cutoff point 
    arr: values for remaining cutoffs
    '''
    return np.log(np.diff(np.append(anchor,arr.reshape(-1))))
#----------------------------------------------------------
# Fit the model and run the regression
#----------------------------------------------------------
def panelFits(Y=None, X=None, FEi=None,FEt=None,FE=None, N=None, T=None, Nold=None, Told=None, 
                dimX=None,dummyIndicator=None,
                model=None, priorVersion=None,  priorLag=0, algorithm="JML", ebc=False,
                sv=None, silent=False,ape_compute=False,seps=0,cutoff0=None,
                mcmc_iters=25000,mcmc_burnin=5000,mcmc_skipsize=2,mcmc_timer=1,mcmc_sv_mle=True,
                para_sd=None,block_size=None):
    #----------------------------------------------------------
    # Initializations
    #----------------------------------------------------------
    # [> Global variable <]
    global parameterUpdate
    global globalkk
    global globaljj
    # [> Epsilon of numerical Python floating number <]
    FLOAT_EPS   = np.finfo(float).eps
    # [> Organize parameters and linear predictors <]
    def getLinprd(paras):
        if model == "logit" or model == "probit":
            thetas = paras[N+T-1:].reshape(-1,1)
        if model =="ologit": 
            thetas = paras[N+T-1:N+T-1+dimX].reshape(-1,1)
        #XB = X@thetas + np.matmul(FEi,paras[0:N-1].reshape(-1,1))+np.matmul(FEt,paras[N-1:N+T-1].reshape(-1,1)) 
        XB = X@thetas + (np.tile(np.concatenate((np.zeros(1),paras[0:N-1])).reshape(1,N),(T,1)).reshape(T,N).T
                         +np.tile(paras[N-1:N+T-1].reshape(1,T),(N,1)) ).reshape(-1,1)
        return thetas,XB
    # [> Get the PDF and CDF functions <]
    def getPdf(arg):
        if model == "logit" or model == "ologit":
            #ret = stats.logistic._pdf(arg)
            cdf = sp.special.expit(arg)
            ret = cdf-cdf**2
        if model =="probit": 
            ret = stats.norm._pdf(arg)
        if model == "mlogit":
            ret = np.column_stack((np.ones(len(arg)), np.exp(arg)))
        return ret
    def getCdf(arg):
        if model == "logit" or model == "ologit":
            ret = sp.special.expit(arg)
        if model =="probit": 
            ret = stats.norm._cdf(arg)
        if model == "mlogit":
            eXB = getPdf(arg)
            ret = eXB/eXB.sum(1)[:,None]
        return ret
    # [> Largest level of dependent variable <]
    Ymax = int(np.max(Y))
    # [> Variable initialization for mlogit <]
    if model == "mlogit":
        bigX = np.hstack((FE,X))
        dimBigX = np.shape(bigX)[1]
        dimX = np.shape(X)[1]
        Ydummies = np.asarray(get_dummies(Y.reshape(-1), drop_first=False))
    #----------------------------------------------------------
    # Objective function
    #----------------------------------------------------------
    def panelNegLogLike(paras):
        if model !="mlogit": thetas,XB = getLinprd(paras)
        if model =="ologit": 
            thresh = np.concatenate((np.zeros(1)+cutoff0,np.exp(paras[-(Ymax-1):].reshape(-1)))).cumsum(axis=0)
            thresh = np.concatenate(([-np.inf], thresh,[np.inf]))
        # --- Compute Log likelihood ---
        if model == "logit" or model == "probit":  
            q=2*Y-1
            #logLike=np.sum(np.log(np.clip(getCdf(q*XB),FLOAT_EPS,1) ))
            logLike=np.sum(np.log(np.maximum(getCdf(q*XB),FLOAT_EPS) ))
        if model =="ologit": 
            low = thresh[Y.astype('i')] - XB
            upp = thresh[Y.astype('i')+1] - XB
            cdf_low = getCdf(low)
            cdf_upp = getCdf(upp)
            cdf_upplow = np.maximum(cdf_upp - cdf_low,FLOAT_EPS)
            logLike=(np.log(cdf_upplow)).sum()
        if model =="mlogit":
            pr = getCdf(np.dot(bigX,paras.reshape(dimBigX, -1, order='F')))
            L = pr[:,1:]
            logLike = np.sum(Ydummies * np.log(pr))
        # --- Compute hessian ---
        if priorVersion is not None:
            if model == "logit": 
                L = getCdf(XB) 
                der1p  = (Y-L).reshape(N,T)
                der2pp = -(L-L**2).reshape(N,T)
            if model == "probit":  
                L = q*getPdf(q*XB)/np.clip(getCdf(q*XB), FLOAT_EPS, 1 - FLOAT_EPS)
                der1p  = L.reshape(N,T)
                der2pp = -(der1p*(der1p+XB.reshape(N,T)))
            if model =="ologit": 
                pdf_low = cdf_low*(1-cdf_low)
                pdf_upp = cdf_upp*(1-cdf_upp)
                pdf_upplow = pdf_upp - pdf_low
                derpdf_low = cdf_low-3*cdf_low**2+2*cdf_low**3
                derpdf_upp = cdf_upp-3*cdf_upp**2+2*cdf_upp**3
                derpdf_upplow = derpdf_upp - derpdf_low
                der1p = -(pdf_upplow/cdf_upplow).reshape(N,T)
                der2pp = (derpdf_upplow/cdf_upplow - (pdf_upplow/cdf_upplow)**2).reshape(N,T)
            if model !="mlogit": diagNegH=np.maximum(-np.append((der2pp).sum(1)[1:],(der2pp).sum(0)),FLOAT_EPS)
            if model =="mlogit": 
                #der2ppdiags = -(L*(1-L)).reshape(N,T,Ymax)
                #diagNegH = -np.concatenate((der2ppdiags[1:,:,:].sum(1),der2ppdiags.sum(0))).reshape(-1)
                partials=[]
                for i in range(Ymax):
                    for j in range(Ymax): 
                        if i == j:
                            partials.append(-(pr[:,i+1]*(1-pr[:,j+1])).reshape(N,T))
                        else:
                            partials.append(-(pr[:,i+1]*( -pr[:,j+1])).reshape(N,T))
                der2pp = (np.array(partials)).transpose(1,2,0)
                diagNegH =  np.concatenate(( np.linalg.det(-der2pp.sum(1).reshape(N,Ymax,Ymax))[1:],
                                             np.linalg.det(-der2pp.sum(0).reshape(T,Ymax,Ymax))  ))
        # --- Compute HAC Score Outer Prod ---
            if priorVersion is not None and priorVersion !="StaticExp":
                if model =="mlogit": 
                    mat=np.linalg.inv(np.concatenate((-der2pp.sum(1)[1:,:].reshape(N-1,Ymax,Ymax),-der2pp.sum(0).reshape(T,Ymax,Ymax))))
                    invdiagNegH = sp.linalg.block_diag(*mat)
                # --- Compute HAC Score Outer Prod (Lag0) ---
                #diagSS  = np.append((der1p**2).sum(1)[1:],(der1p**2).sum(0))
                if model !="mlogit": diagSS  = np.concatenate(((der1p**2).sum(1)[1:],(der1p**2).sum(0)))
                if model =="mlogit": 
                    der1p = (Ydummies[:,1:] - L ).reshape(N,T,Ymax)
                    #diagSS  = np.concatenate(((der1p**2).sum(1)[1:],(der1p**2).sum(0))).reshape(-1)
                    temps = np.tensordot(der1p[:,:,None].transpose(0,1,3,2),der1p[:,:,None], axes=[3,2])
                    term1 = np.diagonal(np.diagonal(temps,axis1=0,axis2=3),axis1=0,axis2=2).sum(3).transpose(2,0,1)
                    term2 = np.diagonal(np.diagonal(temps,axis1=1,axis2=4),axis1=0,axis2=2).sum(3).transpose(2,0,1)
                    diagSS = sp.linalg.block_diag(*(np.concatenate((term1[1:,:,:],term2))))
                # --- Compute HAC Score Outer Prod (Lag1) ---
                if priorLag >= 1:
                    if model !="mlogit": diagSSLag =np.concatenate((2*T/(T-1)*(  (der1p[:,:-1]*der1p[:,1:])[1:,:]  ).sum(axis=1),np.zeros(T)))
                    if model =="mlogit": 
                        #diagSSLag =np.concatenate((2*T/(T-1)*(  (der1p[:,:-1,:]*der1p[:,1:,:])[1:,:,:]  ).sum(axis=1),np.zeros((T,Ymax)))).reshape(-1)
                        temps = np.tensordot(der1p[:,:-1,None].transpose(0,1,3,2),der1p[:,1:,None], axes=[3,2])
                        term1 = np.diagonal(np.diagonal(temps,axis1=0,axis2=3),axis1=0,axis2=2).sum(3).transpose(2,0,1)
                        diagSSLag = 2*T/(T-1)*sp.linalg.block_diag(*(np.concatenate((term1[1:,:,:],np.zeros((T,Ymax,Ymax))))))
                # --- Compute HAC Score Outer Prod (Lag2+) ---
                if priorLag > 1:
                    for ll in range(2,priorLag+1):
                        if model !="mlogit": diagSSLag = diagSSLag+ np.concatenate((2*T/(T-ll)*(  (der1p[:,:-ll]*der1p[:,ll:])[1:,:]  ).sum(axis=1),np.zeros(T)))
                        if model =="mlogit": 
                            #diagSSLag = diagSSLag+ np.concatenate((2*T/(T-ll)*(  (der1p[:,:-ll,:]*der1p[:,ll:])[1:,:,:]  ).sum(axis=1),np.zeros((T,Ymax)))).reshape(-1)
                            temps = np.tensordot(der1p[:,:-ll,None].transpose(0,1,3,2),der1p[:,ll:,None], axes=[3,2])
                            term1 = np.diagonal(np.diagonal(temps,axis1=0,axis2=3),axis1=0,axis2=2).sum(3).transpose(2,0,1)
                            diagSSLag = diagSSLag+ 2*T/(T-ll)*sp.linalg.block_diag(*(np.concatenate((term1[1:,:,:],np.zeros((T,Ymax,Ymax))))))
        # --- Different Versions of Priors ---
        if priorVersion is None: prior = 0
        if priorVersion =="StaticExp": 
            prior = 0.5*np.log(diagNegH).sum()
        if priorVersion == "Generic" and priorLag==0:
            if model !="mlogit": prior = -0.5*((diagSS)/diagNegH).sum()
            if model =="mlogit": prior = -0.5*(np.diag(invdiagNegH@diagSS).sum())
        if priorVersion == "Generic" and priorLag>=1:
            if model !="mlogit": prior = -0.5*((diagSSLag+diagSS)/diagNegH).sum()
            if model =="mlogit": prior = -0.5*(np.diag(invdiagNegH@(diagSSLag+diagSS)).sum())  
        if priorVersion == "Binary" and priorLag==0:
            prior = 0.5*np.log(diagNegH).sum()
        if priorVersion == "Binary" and priorLag>=1:
            if model !="mlogit": prior = -0.5*((diagSSLag)/diagNegH)[:N-1].sum() + 0.5*(np.log(diagNegH).sum())
            if model =="mlogit": prior = -0.5*(np.diag(invdiagNegH@(diagSSLag)).sum())  + 0.5*(np.log(diagNegH).sum())
        if priorVersion is not None and algorithm == "MCMC": prior=prior+0.5*np.log(diagNegH).sum()
        return -(logLike+prior)
    def panelNegLogLike_fixTheta(parasFixTheta):
        paras = np.append(parasFixTheta,estTheta)
        thetas,XB = getLinprd(paras)
        # --- Compute Log likelihood ---
        if model == "logit" or model == "probit":  
            q=2*Y-1
            #logLike=np.sum(np.log(np.clip(getCdf(q*XB),FLOAT_EPS,1) ))
            logLike=np.sum(np.log(np.maximum(getCdf(q*XB),FLOAT_EPS) ))
        # --- Compute hessian ---
        if priorVersion is not None:
            if model == "logit": 
                L = getCdf(XB) 
                der1p  = (Y-L).reshape(N,T)
                der2pp = -(L-L**2).reshape(N,T)
            if model == "probit":  
                L = q*getPdf(q*XB)/np.clip(getCdf(q*XB), FLOAT_EPS, 1 - FLOAT_EPS)
                der1p  = L.reshape(N,T)
                der2pp = -(der1p*(der1p+XB.reshape(N,T)))
            diagNegH=np.maximum(-np.append((der2pp).sum(1)[1:],(der2pp).sum(0)),FLOAT_EPS)
        # --- Compute HAC Score Outer Prod ---
            if priorVersion is not None and priorVersion !="StaticExp":
                # --- Compute HAC Score Outer Prod (Lag0) ---
                #diagSS  = np.append((der1p**2).sum(1)[1:],(der1p**2).sum(0))
                diagSS  = np.concatenate(((der1p**2).sum(1)[1:],(der1p**2).sum(0)))
                # --- Compute HAC Score Outer Prod (Lag1) ---
                if priorLag >= 1:
                    diagSSLag =np.concatenate((2*T/(T-1)*(  (der1p[:,:-1]*der1p[:,1:])[1:,:]  ).sum(axis=1),np.zeros(T)))
                # --- Compute HAC Score Outer Prod (Lag2+) ---
                if priorLag > 1:
                    for ll in range(2,priorLag+1):
                        diagSSLag = diagSSLag+ np.concatenate((2*T/(T-ll)*(  (der1p[:,:-ll]*der1p[:,ll:])[1:,:]  ).sum(axis=1),np.zeros(T)))
        # --- Different Versions of Priors ---
        if priorVersion is None: prior = 0
        if priorVersion =="StaticExp": 
            prior = 0.5*np.log(diagNegH).sum()
        if priorVersion == "Generic" and priorLag==0:
            prior = -0.5*((diagSS)/diagNegH).sum()
        if priorVersion == "Generic" and priorLag>=1:
            prior = -0.5*((diagSSLag+diagSS)/diagNegH).sum()
        if priorVersion == "Binary" and priorLag==0:
            prior = 0.5*np.log(diagNegH).sum()
        if priorVersion == "Binary" and priorLag>=1:
            prior = -0.5*((diagSSLag)/diagNegH)[:N-1].sum() + 0.5*(np.log(diagNegH).sum())
        if priorVersion is not None and algorithm == "MCMC": prior=prior+0.5*np.log(diagNegH).sum()
        return -(logLike+prior)
    # Objective functions - optimization with concentration scheme 
    def panelNegLogLikeConcen(parameterSlice):
        global parameterUpdate
        parameterUpdate[newindices[start_i:end_i]] = parameterSlice
        if model !="mlogit": thetas,XB = getLinprd(paras)
        if model =="ologit": 
            thresh = np.concatenate((np.zeros(1)+cutoff0,np.exp(paras[-(Ymax-1):].reshape(-1)))).cumsum(axis=0)
            thresh = np.concatenate(([-np.inf], thresh,[np.inf]))
        # --- Compute Log likelihood ---
        if model == "logit" or model == "probit":  
            q=2*Y-1
            #logLike=np.sum(np.log(np.clip(getCdf(q*XB),FLOAT_EPS,1) ))
            logLike=np.sum(np.log(np.maximum(getCdf(q*XB),FLOAT_EPS) ))
        if model =="ologit": 
            low = thresh[Y.astype('i')] - XB
            upp = thresh[Y.astype('i')+1] - XB
            cdf_low = getCdf(low)
            cdf_upp = getCdf(upp)
            cdf_upplow = np.maximum(cdf_upp - cdf_low,FLOAT_EPS)
            logLike=(np.log(cdf_upplow)).sum()
        if model =="mlogit":
            pr = getCdf(np.dot(bigX,paras.reshape(dimBigX, -1, order='F')))
            L = pr[:,1:]
            logLike = np.sum(Ydummies * np.log(pr))
        # --- Compute hessian ---
        if priorVersion is not None:
            if model == "logit": 
                L = getCdf(XB) 
                der1p  = (Y-L).reshape(N,T)
                der2pp = -(L-L**2).reshape(N,T)
            if model == "probit":  
                L = q*getPdf(q*XB)/np.clip(getCdf(q*XB), FLOAT_EPS, 1 - FLOAT_EPS)
                der1p  = L.reshape(N,T)
                der2pp = -(der1p*(der1p+XB.reshape(N,T)))
            if model =="ologit": 
                pdf_low = cdf_low*(1-cdf_low)
                pdf_upp = cdf_upp*(1-cdf_upp)
                pdf_upplow = pdf_upp - pdf_low
                derpdf_low = cdf_low-3*cdf_low**2+2*cdf_low**3
                derpdf_upp = cdf_upp-3*cdf_upp**2+2*cdf_upp**3
                derpdf_upplow = derpdf_upp - derpdf_low
                der1p = -(pdf_upplow/cdf_upplow).reshape(N,T)
                der2pp = (derpdf_upplow/cdf_upplow - (pdf_upplow/cdf_upplow)**2).reshape(N,T)
            if model !="mlogit": diagNegH=np.maximum(-np.append((der2pp).sum(1)[1:],(der2pp).sum(0)),FLOAT_EPS)
            if model =="mlogit": 
                der2ppdiags = -(L*(1-L)).reshape(N,T,Ymax)
                diagNegH = -np.append(der2ppdiags.sum(1)[1:,:].reshape(-1),der2ppdiags.sum(0).reshape(-1))
        # --- Compute HAC Score Outer Prod ---
            if priorVersion is not None and priorVersion !="StaticExp":
                # --- Compute HAC Score Outer Prod (Lag0) ---
                #diagSS  = np.append((der1p**2).sum(1)[1:],(der1p**2).sum(0))
                if model !="mlogit": diagSS  = np.concatenate(((der1p**2).sum(1)[1:],(der1p**2).sum(0)))
                if model =="mlogit": 
                    der1p = (Ydummies[:,1:] - L ).reshape(N,T,Ymax)
                    diagSS  = np.concatenate(((der1p**2).sum(1)[1:],(der1p**2).sum(0))).reshape(-1)
                # --- Compute HAC Score Outer Prod (Lag1) ---
                if priorLag >= 1:
                    if model !="mlogit": diagSSLag =np.concatenate((2*T/(T-1)*(  (der1p[:,:-1]*der1p[:,1:])[1:,:]  ).sum(axis=1),np.zeros(T)))
                    if model =="mlogit": diagSSLag =np.concatenate((2*T/(T-1)*(  (der1p[:,:-1,:]*der1p[:,1:,:])[1:,:,:]  ).sum(axis=1),np.zeros((T,Ymax)))).reshape(-1)
                # --- Compute HAC Score Outer Prod (Lag2+) ---
                if priorLag > 1:
                    for ll in range(2,priorLag+1):
                        if model !="mlogit": diagSSLag = diagSSLag+ np.concatenate((2*T/(T-ll)*(  (der1p[:,:-ll]*der1p[:,ll:])[1:,:]  ).sum(axis=1),np.zeros(T)))
                        if model =="mlogit": diagSSLag = diagSSLag+ np.concatenate((2*T/(T-ll)*(  (der1p[:,:-ll,:]*der1p[:,ll:])[1:,:,:]  ).sum(axis=1),np.zeros((T,Ymax)))).reshape(-1)
        # --- Different Versions of Priors ---
        if priorVersion is None: prior = 0
        if priorVersion =="StaticExp": 
            prior = 0.5*np.log(diagNegH).sum()
        if priorVersion == "Generic" and priorLag==0:
            prior = -0.5*((diagSS)/diagNegH).sum()
        if priorVersion == "Generic" and priorLag>=1:
            prior = -0.5*((diagSSLag+diagSS)/diagNegH).sum()
        if priorVersion == "Binary" and priorLag==0:
            prior = 0.5*np.log(diagNegH).sum()
        if priorVersion == "Binary" and priorLag>=1:
            prior = -0.5*((diagSSLag)/diagNegH)[:N-1].sum() + 0.5*(np.log(diagNegH).sum())
        return -(logLike+prior) 
    #---------------------------------------------------------------------------------------------------------
    # Negative Hessian matrix 
    #---------------------------------------------------------------------------------------------------------
    def panelNegHess(paras): 
        if model !="mlogit": thetas,XB = getLinprd(paras)
        if model =="ologit": 
            thresh = np.concatenate(([-np.inf], np.zeros(1)+cutoff0, paras[-(Ymax-1):].reshape(-1),[np.inf]))
        bigX = np.hstack((FE,X))
        if model == "logit":
            L = getCdf(XB) 
            negHess_all = -np.dot((-L*(1-L)*bigX).T,bigX)
        if model == "probit":
            q=2*Y-1 
            L = q*getPdf(q*XB)/np.clip(getCdf(q*XB), FLOAT_EPS, 1 - FLOAT_EPS)
            negHess_all = -np.dot((-L*(L+XB)*bigX).T,bigX) 
        if model =="ologit": 
            low = thresh[Y.astype('i')] - XB
            upp = thresh[Y.astype('i')+1] - XB
            cdf_low = getCdf(low)
            cdf_upp = getCdf(upp)
            cdf_upplow = np.maximum(cdf_upp - cdf_low,FLOAT_EPS)
            pdf_low = getPdf(low)
            pdf_upp = getPdf(upp)
            pdf_upplow = pdf_upp - pdf_low
            derpdf_low = cdf_low-3*cdf_low**2+2*cdf_low**3
            derpdf_upp = cdf_upp-3*cdf_upp**2+2*cdf_upp**3
            derpdf_upplow = derpdf_upp - derpdf_low
            negHess_all = np.zeros((N+T-1+dimX+Ymax-1,N+T-1+dimX+Ymax-1))
            negHess_all[:N+T-1+dimX,:N+T-1+dimX] = -np.dot(((derpdf_upplow/cdf_upplow - (pdf_upplow/cdf_upplow)**2)*bigX).T,bigX)
            cutoff_secondderA =  (derpdf_upp/cdf_upplow - (pdf_upp/cdf_upplow)**2)
            cutoff_secondderB = (-derpdf_low/cdf_upplow - (pdf_low/cdf_upplow)**2)
            cutoff_crossA = -( derpdf_upp/cdf_upplow - pdf_upplow*  pdf_upp /(cdf_upplow**2) )*bigX
            cutoff_crossB = -(-derpdf_low/cdf_upplow - pdf_upplow*(-pdf_low)/(cdf_upplow**2) )*bigX
            for jj in range(1,Ymax):
                negHess_all[N+T-1+dimX+jj-1,N+T-1+dimX+jj-1] = -cutoff_secondderA[Y==jj].sum()-cutoff_secondderB[Y==jj+1].sum()
                negHess_all[N+T-1+dimX+jj-1,:N+T-1+dimX] = -(cutoff_crossA[Y.reshape(-1)==jj,:].sum(axis=0)+cutoff_crossB[Y.reshape(-1)==jj+1,:].sum(axis=0))
                negHess_all[:N+T-1+dimX,N+T-1+dimX+jj-1]= negHess_all[N+T-1+dimX+jj-1,:N+T-1+dimX] 
                if jj!=1: 
                    negHess_all[-jj,-jj+1] = -(pdf_upp*pdf_low/cdf_upplow**2)[Y==jj].sum()
                    negHess_all[-jj+1,-jj] = negHess_all[-jj,-jj+1]
        if model =="mlogit":
            pr = getCdf(np.dot(bigX,paras.reshape(dimBigX, -1, order='F')))
            partials = []
            for i in range(Ymax):
                for j in range(Ymax): 
                    if i == j:
                        partials.append(-np.dot(((pr[:,i+1]*(1-pr[:,j+1]))[:,None]*bigX).T,bigX))
                    else:
                        partials.append(-np.dot(((pr[:,i+1]*( -pr[:,j+1]))[:,None]*bigX).T,bigX))
            H = np.array(partials)
            Hflat = (np.transpose(H.reshape(Ymax, Ymax, dimBigX, dimBigX), (0, 2, 1, 3)).reshape( 
                     (Ymax)*dimBigX, (Ymax)*dimBigX ) )
            negHess_all = -Hflat
        return negHess_all
    #----------------------------------------------------------
    # Negative Jacobian/Score
    #----------------------------------------------------------
    def panelNegScore(paras):     
        if model !="mlogit":
            der1p,der1b,der2pp,der2pb,der3ppp,der3ppb=panelHelper(paras)
            score = np.concatenate((der1p.sum(1)[1:],der1p.sum(0),der1b.sum(0).sum(0)))
            if priorVersion is not None: 
                fenmu1 = der2pp.sum(1)
                fenmu0 = der2pp.sum(0)
                # --- Compute various terms used in optimization ---
                if priorVersion =="StaticExp" or priorVersion =="Binary":
                    sPa = (  (der3ppp.sum(1)/fenmu1) + (der3ppp/(fenmu0.reshape(1,-1))).sum(1)  )[1:]/2
                    sPg = (  (der3ppp/fenmu1.reshape(-1,1))[1:].sum(0) + der3ppp.sum(0)/fenmu0  )/2
                    sPb = (  (der3ppb.sum(1)/fenmu1.reshape(-1,1))[1:].sum(0) + (der3ppb.sum(0)/fenmu0.reshape(-1,1)).sum(0)  )/2
                if priorVersion == "Generic":
                    sPa=  (   (der2pp*der1p).sum(1) / fenmu1  
                            -  0.5*(  (der3ppp.sum(1)) * ((der1p**2).sum(1))  ) / (fenmu1**2)
                            +  ((der2pp*der1p)/(fenmu0.reshape(1,-1))).sum(1) 
                            -  0.5*(  ( der3ppp*((der1p**2).sum(0)) ) / ((fenmu0**2).reshape(1,-1))  ).sum(1) 
                          )[1:]
                    sPg=  (   ( (der2pp*der1p) / fenmu1.reshape(-1,1)  )[1:].sum(0)
                            -  0.5*(  ( der3ppp*((der1p**2).sum(1).reshape(-1,1)) ) / ((fenmu1**2).reshape(-1,1))  )[1:].sum(0)
                            +  (der2pp*der1p).sum(0) / fenmu0
                            -  0.5*(  (der3ppp.sum(0)) * ((der1p**2).sum(0))  ) / (fenmu0**2)
                          )
                    sPb=  (   ( (der2pb*der1p.reshape(N,T,1)).sum(1) / fenmu1.reshape(-1,1) )[1:].sum(0)
                            - 0.5* (  der3ppb.sum(1)*((der1p**2).sum(1)).reshape(-1,1) / (fenmu1**2).reshape(-1,1)  )[1:].sum(0)        
                            + ( (der2pb*der1p.reshape(N,T,1)).sum(0) / fenmu0.reshape(-1,1) ).sum(0)
                            - 0.5* (  der3ppb.sum(0)*((der1p**2).sum(0)).reshape(-1,1) / (fenmu0**2).reshape(-1,1)  ).sum(0)   
                          )
                if priorLag > 0 and priorVersion !="StaticExp":
                    for ll in range(1,priorLag+1):
                        sPaL=(   (der2pp[:,:-ll]*der1p[:,ll: ]).sum(1) / fenmu1  
                                + (der2pp[:,ll: ]*der1p[:,:-ll]).sum(1) / fenmu1  
                                - (  (der3ppp.sum(1)) * ((der1p[:,:-ll]*der1p[:,ll:]).sum(1))  ) / (fenmu1**2)
                              )[1:]
                        sPbL=(   ( ( der2pb[:,:-ll,:]*((der1p.reshape(N,T,1))[:,ll: ,:]) ).sum(1) / fenmu1.reshape(-1,1) )[1:].sum(0)
                                + ( ( der2pb[:,ll: ,:]*((der1p.reshape(N,T,1))[:,:-ll,:]) ).sum(1) / fenmu1.reshape(-1,1) )[1:].sum(0)
                                - (  (der3ppb.sum(1)) * ((der1p[:,:-ll]*der1p[:,ll:]).sum(1)).reshape(-1,1) / (fenmu1**2).reshape(-1,1)  )[1:].sum(0)  
                              )
                        sPgL=(    ( np.hstack((
                                          (der2pp[:,:ll]*der1p[:,ll:2*ll]),
                                          (der1p[:,:-2*ll]*der2pp[:,ll:-ll])+(der2pp[:,ll:-ll]*der1p[:,2*ll:]),
                                          (der1p[:,-2*ll:-ll]*der2pp[:,-ll:])
                                       )) / fenmu1.reshape(-1,1)  )[1:,:].sum(0) 
                                -  (  ( der3ppp*((der1p[:,:-ll]*der1p[:,ll:]).sum(1).reshape(-1,1)) ) / ((fenmu1**2).reshape(-1,1))  )[1:,:].sum(0)
                              )
                        sPa = sPa + sPaL*T/(T-ll)
                        sPg = sPg + sPgL*T/(T-ll)
                        sPb = sPb + sPbL*T/(T-ll)
                score= score+np.concatenate((sPa,sPg,sPb))
        if model =="mlogit":
            firstterm = Ydummies[:,1:] - getCdf(np.dot(bigX,paras.reshape(dimBigX, -1, order='F')))[:,1:]
            score= np.dot(firstterm.T, bigX).flatten()
        return -score
    def panelNegScore_fixTheta(parasFixTheta):     
        paras = np.append(parasFixTheta,estTheta)
        der1p,der1b,der2pp,der2pb,der3ppp,der3ppb=panelHelper(paras)
        score = np.concatenate((der1p.sum(1)[1:],der1p.sum(0)))
        if priorVersion is not None: 
            fenmu1 = der2pp.sum(1)
            fenmu0 = der2pp.sum(0)
            # --- Compute various terms used in optimization ---
            if priorVersion =="StaticExp" or priorVersion =="Binary":
                sPa = (  (der3ppp.sum(1)/fenmu1) + (der3ppp/(fenmu0.reshape(1,-1))).sum(1)  )[1:]/2
                sPg = (  (der3ppp/fenmu1.reshape(-1,1))[1:].sum(0) + der3ppp.sum(0)/fenmu0  )/2
            if priorVersion == "Generic":
                sPa=  (   (der2pp*der1p).sum(1) / fenmu1  
                        -  0.5*(  (der3ppp.sum(1)) * ((der1p**2).sum(1))  ) / (fenmu1**2)
                        +  ((der2pp*der1p)/(fenmu0.reshape(1,-1))).sum(1) 
                        -  0.5*(  ( der3ppp*((der1p**2).sum(0)) ) / ((fenmu0**2).reshape(1,-1))  ).sum(1) 
                      )[1:]
                sPg=  (   ( (der2pp*der1p) / fenmu1.reshape(-1,1)  )[1:].sum(0)
                        -  0.5*(  ( der3ppp*((der1p**2).sum(1).reshape(-1,1)) ) / ((fenmu1**2).reshape(-1,1))  )[1:].sum(0)
                        +  (der2pp*der1p).sum(0) / fenmu0
                        -  0.5*(  (der3ppp.sum(0)) * ((der1p**2).sum(0))  ) / (fenmu0**2)
                      )
            if priorLag > 0 and priorVersion !="StaticExp":
                for ll in range(1,priorLag+1):
                    sPaL=(   (der2pp[:,:-ll]*der1p[:,ll: ]).sum(1) / fenmu1  
                            + (der2pp[:,ll: ]*der1p[:,:-ll]).sum(1) / fenmu1  
                            - (  (der3ppp.sum(1)) * ((der1p[:,:-ll]*der1p[:,ll:]).sum(1))  ) / (fenmu1**2)
                          )[1:]
                    sPgL=(    ( np.hstack((
                                      (der2pp[:,:ll]*der1p[:,ll:2*ll]),
                                      (der1p[:,:-2*ll]*der2pp[:,ll:-ll])+(der2pp[:,ll:-ll]*der1p[:,2*ll:]),
                                      (der1p[:,-2*ll:-ll]*der2pp[:,-ll:])
                                   )) / fenmu1.reshape(-1,1)  )[1:,:].sum(0) 
                            -  (  ( der3ppp*((der1p[:,:-ll]*der1p[:,ll:]).sum(1).reshape(-1,1)) ) / ((fenmu1**2).reshape(-1,1))  )[1:,:].sum(0)
                          )
                    sPa = sPa + sPaL*T/(T-ll)
                    sPg = sPg + sPgL*T/(T-ll)
            score= score+np.concatenate((sPa,sPg))
        return -score
    def panelNegScoreConcen(parameterSlice):
        global parameterUpdate
        parameterUpdate[newindices[start_i:end_i]] = parameterSlice
        der1p,der1b,der2pp,der2pb,der3ppp,der3ppb=panelHelper(parameterUpdate)
        score = np.concatenate((der1p.sum(1)[1:],der1p.sum(0),der1b.sum(0).sum(0)))
        if priorVersion is not None: 
            fenmu1 = der2pp.sum(1)
            fenmu0 = der2pp.sum(0)
            # --- Compute various terms used in optimization ---
            if priorVersion =="StaticExp" or priorVersion =="Binary":
                sPa = (  (der3ppp.sum(1)/fenmu1) + (der3ppp/(fenmu0.reshape(1,-1))).sum(1)  )[1:]/2
                sPg = (  (der3ppp/fenmu1.reshape(-1,1))[1:].sum(0) + der3ppp.sum(0)/fenmu0  )/2
                sPb = (  (der3ppb.sum(1)/fenmu1.reshape(-1,1))[1:].sum(0) + (der3ppb.sum(0)/fenmu0.reshape(-1,1)).sum(0)  )/2
            if priorVersion == "Generic":
                sPa=  (   (der2pp*der1p).sum(1) / fenmu1  
                        -  0.5*(  (der3ppp.sum(1)) * ((der1p**2).sum(1))  ) / (fenmu1**2)
                        +  ((der2pp*der1p)/(fenmu0.reshape(1,-1))).sum(1) 
                        -  0.5*(  ( der3ppp*((der1p**2).sum(0)) ) / ((fenmu0**2).reshape(1,-1))  ).sum(1) 
                      )[1:]
                sPg=  (   ( (der2pp*der1p) / fenmu1.reshape(-1,1)  )[1:].sum(0)
                        -  0.5*(  ( der3ppp*((der1p**2).sum(1).reshape(-1,1)) ) / ((fenmu1**2).reshape(-1,1))  )[1:].sum(0)
                        +  (der2pp*der1p).sum(0) / fenmu0
                        -  0.5*(  (der3ppp.sum(0)) * ((der1p**2).sum(0))  ) / (fenmu0**2)
                      )
                sPb=  (   ( (der2pb*der1p.reshape(N,T,1)).sum(1) / fenmu1.reshape(-1,1) )[1:].sum(0)
                        - 0.5* (  der3ppb.sum(1)*((der1p**2).sum(1)).reshape(-1,1) / (fenmu1**2).reshape(-1,1)  )[1:].sum(0)        
                        + ( (der2pb*der1p.reshape(N,T,1)).sum(0) / fenmu0.reshape(-1,1) ).sum(0)
                        - 0.5* (  der3ppb.sum(0)*((der1p**2).sum(0)).reshape(-1,1) / (fenmu0**2).reshape(-1,1)  ).sum(0)   
                      )
            if priorLag > 0 and priorVersion !="StaticExp":
                for ll in range(1,priorLag+1):
                    sPaL=(   (der2pp[:,:-ll]*der1p[:,ll: ]).sum(1) / fenmu1  
                            + (der2pp[:,ll: ]*der1p[:,:-ll]).sum(1) / fenmu1  
                            - (  (der3ppp.sum(1)) * ((der1p[:,:-ll]*der1p[:,ll:]).sum(1))  ) / (fenmu1**2)
                          )[1:]
                    sPbL=(   ( ( der2pb[:,:-ll,:]*((der1p.reshape(N,T,1))[:,ll: ,:]) ).sum(1) / fenmu1.reshape(-1,1) )[1:].sum(0)
                            + ( ( der2pb[:,ll: ,:]*((der1p.reshape(N,T,1))[:,:-ll,:]) ).sum(1) / fenmu1.reshape(-1,1) )[1:].sum(0)
                            - (  (der3ppb.sum(1)) * ((der1p[:,:-ll]*der1p[:,ll:]).sum(1)).reshape(-1,1) / (fenmu1**2).reshape(-1,1)  )[1:].sum(0)  
                          )
                    sPgL=(    ( np.hstack((
                                      (der2pp[:,:ll]*der1p[:,ll:2*ll]),
                                      (der1p[:,:-2*ll]*der2pp[:,ll:-ll])+(der2pp[:,ll:-ll]*der1p[:,2*ll:]),
                                      (der1p[:,-2*ll:-ll]*der2pp[:,-ll:])
                                   )) / fenmu1.reshape(-1,1)  )[1:,:].sum(0) 
                            -  (  ( der3ppp*((der1p[:,:-ll]*der1p[:,ll:]).sum(1).reshape(-1,1)) ) / ((fenmu1**2).reshape(-1,1))  )[1:,:].sum(0)
                          )
                    sPa = sPa + sPaL*T/(T-ll)
                    sPg = sPg + sPgL*T/(T-ll)
                    sPb = sPb + sPbL*T/(T-ll)
            score= score+np.concatenate((sPa,sPg,sPb))
        return -score 
    #----------------------------------------------------------
    # Functions for APE
    #----------------------------------------------------------
    # Calculate the APE
    def panelApeCalc(paras):
        if model != "mlogit": thetas,XB = getLinprd(paras)
        if model == "ologit": 
            thresh = np.concatenate(([-np.inf], np.zeros(1)+cutoff0, paras[-(Ymax-1):].reshape(-1),[np.inf]))
        if model == "logit" or model == "probit":  
            ape_mat = (thetas.reshape(-1)*getPdf(XB)).reshape((N,T,dimX))
            if seps==1:
                ape = (ape_mat.sum(axis=0).sum(axis=0) )/(Nold*Told)
            else:
                ape = ape_mat.mean(axis=0).mean(axis=0)
            for kx in range(dimX):
                if dummyIndicator[kx]==True: 
                    X0B = XB - X[:,kx].reshape(-1,1)*thetas[kx]
                    X1B = X0B + np.ones((N*T,1))*thetas[kx]
                    ape_mat[:,:,kx] = (getCdf(X1B)-getCdf(X0B)).reshape(N,T)
                    ape[kx] = ((ape_mat[:,:,kx]).sum(axis=0).sum(axis=0))/(Nold*Told)
        if model == "ologit": 
            ape_mat = np.zeros(( dimX,  N, T,  Ymax+1  ))
            #ape     = np.zeros(( dimX,         Ymax+1  ))
            for kk in range(Ymax+1):
                ape_mat[:,:,:,kk] = -(((getPdf(thresh-XB)[:,kk] - getPdf(thresh-XB)[:,kk+1]))*thetas).reshape(dimX,N,T)
                for kx in range(dimX):
                    if dummyIndicator[kx]==True: 
                        X0B = XB - X[:,kx].reshape(-1,1)*thetas[kx]
                        X1B = X0B + np.ones((N*T,1))*thetas[kx]
                        ape_mat[kx,:,:,kk] = (
                                               (getCdf(thresh-X1B)[:,kk] - getCdf(thresh-X1B)[:,kk+1]).reshape(N,T)
                                              -(getCdf(thresh-X0B)[:,kk] - getCdf(thresh-X0B)[:,kk+1]).reshape(N,T)
                                             )
            ape = ape_mat.mean(axis=1).mean(axis=1)
        if model == "mlogit":
            thetas = (paras.reshape(dimBigX, -1, order='F'))[-dimX:,:]
            pr = getCdf(np.dot(bigX,paras.reshape(dimBigX, -1, order='F')))
            L = pr[:,1:]
            bigL = (thetas[:,None]*L).transpose(0,2,1)
            bigLSumj = bigL.sum(1)
            ape_mat = np.zeros(( dimX, Ymax, N, T ))
            for kk in range(dimX):
                if dummyIndicator[kk]==True:
                    bigX0=np.copy(bigX)
                    bigX1=np.copy(bigX)
                    bigX0[:,N+T-1+kk]=0
                    bigX1[:,N+T-1+kk]=1
                    L0 = (getCdf(np.dot(bigX0,paras.reshape(dimBigX, -1, order='F'))))[:,1:]
                    L1 = (getCdf(np.dot(bigX1,paras.reshape(dimBigX, -1, order='F'))))[:,1:]
                    ape_mat[kk,:,:,:] = (L1-L0).reshape(Ymax,N,T)
                else:
                    ape_mat[kk,:,:,:] = (bigL[kk,:,:] - (bigLSumj[kk,:]*L.T)).reshape(Ymax,N,T)
            ape = (ape_mat.sum(2).sum(2))/(Nold*Told)
            ape_mat=ape_mat.transpose(2,3,0,1)
        return ape, ape_mat
    # APE's asymptotic standard errors
    def panelApeSE(paras):
        ape,ape_mat = panelApeCalc(paras)
        first_der_theta,first_der_lambda,_,first_der_lambda_mat=panelApeHelper(paras)
        negHess = panelNegHess(paras)
        if model != "mlogit": 
            invW = np.linalg.inv(negHess)[N+T-1:,N+T-1:]
            invH = np.linalg.inv(negHess[:N+T-1,:N+T-1])
            crossL = -negHess[N+T-1:,:N+T-1]
        if model == "mlogit": 
            invW = np.linalg.inv(negHess).reshape(dimBigX,Ymax,dimBigX,Ymax,order='F')[-dimX:,:,-dimX:,:].reshape(dimX*Ymax, dimX*Ymax, order='F')
            invH = np.linalg.inv(negHess.reshape(dimBigX,Ymax,dimBigX,Ymax,order='F')[:N+T-1,:,:N+T-1,:].reshape((N+T-1)*Ymax, (N+T-1)*Ymax, order='F'))
            crossL = -negHess.reshape(dimBigX,Ymax,dimBigX,Ymax,order='F')[-dimX:,:,:N+T-1,:].reshape(dimX*Ymax, (N+T-1)*Ymax, order='F')
        if model == "logit" or model == "probit":
            term1 = first_der_theta + first_der_lambda.T@invH@crossL.T
            var2 = term1@invW@term1.T + first_der_lambda.T@invH@first_der_lambda 
            #first_der = np.vstack((first_der_lambda,first_der_theta.T))
            #var2 = first_der.T@np.linalg.inv(negHess)@first_der
            var1=np.zeros((dimX,dimX))
            for kx in range(dimX):
                if dummyIndicator[kx]==True: ape[kx] = ape[kx]*(Nold)/N
            ape_mat_tilde = ape_mat - ape
            # slowest - raw version
            #for ii in range(N):
            #    for tt in range(T):
            #        for tau in range(T):
            #            var1 =var1+ ape_mat_tilde[ii,tt,:].reshape((-1,1))@ape_mat_tilde[ii,tau,:].reshape((1,-1))
            #        for jj in range(N):
            #            if jj!=ii: var1 =var1+  ape_mat_tilde[ii,tt,:].reshape((-1,1))@ape_mat_tilde[jj,tt,:].reshape((1,-1))
            # fastest
            term2 = np.matmul(ape_mat_tilde.transpose(2,0,1), ape_mat_tilde.transpose(2,1,0))
            term2=term2.sum(1).sum(1)-np.diagonal(term2,axis1=1,axis2=2).sum(1)
            term1 = np.matmul(ape_mat_tilde.transpose(2,1,0), ape_mat_tilde.transpose(2,0,1))
            term1=term1.sum(1).sum(1)#-np.diagonal(term1,axis1=1,axis2=2).sum(1)
            var1=np.diag(term1+term2)

            var1 = var1/((N*T-dimX-N-T+2)**2)
            #print("var1",np.diag(var1))
            apeSE = np.sqrt(np.diag(var1+var2))
        if model =="mlogit": 
            apeSE = np.zeros((dimX,Ymax))
            for kk in range(Ymax):
                term1 = first_der_theta[:,kk,:] + first_der_lambda[:,kk,:]@invH@crossL.T
                var2 = term1@invW@term1.T+ first_der_lambda[:,kk,:]@invH@first_der_lambda[:,kk,:].T 
                var1=np.zeros((dimX,dimX))
                ape_mat_tilde = ((ape_mat[:,:,:,kk]).T - (ape[:,kk].reshape(dimX,1,1))).T
                term2 = np.matmul(ape_mat_tilde.transpose(2,0,1), ape_mat_tilde.transpose(2,1,0))
                term2=term2.sum(1).sum(1)#-np.diagonal(term2,axis1=1,axis2=2).sum(1)
                term1 = np.matmul(ape_mat_tilde.transpose(2,1,0), ape_mat_tilde.transpose(2,0,1))
                term1=term1.sum(1).sum(1)-np.diagonal(term1,axis1=1,axis2=2).sum(1)
                var1=np.diag(term1+term2)

                var1 = var1/((N*T-dimBigX+1)**2)
                apeSE[:,kk] = np.sqrt(np.diag(var1+var2))
        if model =="ologit": 
            apeSE = np.zeros((dimX,Ymax+1))
            for kk in range(Ymax+1):
                term1 = first_der_theta[:,kk,:] - first_der_lambda[:,kk,:]@invH@negHess[:N+T-1,N+T-1:]
                var2 = term1@invW@term1.T+ first_der_lambda[:,kk,:]@invH@first_der_lambda[:,kk,:].T 
                var1=np.zeros((dimX,dimX))
                ape_mat_tilde = (  (ape_mat[:,:,:,kk]).T - (ape[:,kk])  ).T
                # slowest - raw version
                #for ii in range(N):
                #    for tt in range(T):
                #        for tau in range(T):
                #            var1 =var1+ ape_mat_tilde[:,ii,tt].reshape((-1,1))@ape_mat_tilde[:,ii,tau].reshape((1,-1))
                #        for jj in range(N):
                #            if jj!=ii: var1 =var1+  ape_mat_tilde[:,ii,tt].reshape((-1,1))@ape_mat_tilde[:,jj,tt].reshape((1,-1))
                # fastest
                term2 = np.matmul(ape_mat_tilde.T.transpose(2,0,1), ape_mat_tilde.T.transpose(2,1,0))
                term2=term2.sum(1).sum(1)#-np.diagonal(term2,axis1=1,axis2=2).sum(1)
                term1 = np.matmul(ape_mat_tilde.T.transpose(2,1,0), ape_mat_tilde.T.transpose(2,0,1))
                term1=term1.sum(1).sum(1)-np.diagonal(term1,axis1=1,axis2=2).sum(1)
                var1=np.diag(term1+term2)

                var1 = var1/((N*T)**2)
                apeSE[:,kk] = np.sqrt(np.diag(var1+var2))
        return apeSE
    #----------------------------------------------------------
    # Helper functions for instantly computing various terms
    #----------------------------------------------------------
    # Helper function for computing various partial derivatives of log-likelihood for each obs.
    def panelHelper(paras):
        thetas,XB = getLinprd(paras)
        if model =="ologit": 
            thresh = np.concatenate((np.zeros(1)+cutoff0,np.exp(paras[-(Ymax-1):].reshape(-1)))).cumsum(axis=0)
            thresh = np.concatenate(([-np.inf], thresh,[np.inf]))
        # --- Compute various terms used in optimization ---
        if model == "logit":  
            L = getCdf(XB) 
            der1p  = (Y-L).reshape(N,T)
            der2pp = -(L-L**2).reshape(N,T)
            der3ppp= -(L-3*L**2+2*L**3).reshape(N,T)
        if model == "probit": 
            q=2*Y-1 
            L = q*getPdf(q*XB)/np.clip(getCdf(q*XB), FLOAT_EPS, 1 - FLOAT_EPS)
            der1p  = L.reshape(N,T)
            der2pp = -(der1p*(der1p+XB.reshape(N,T)))
            der2pp = -np.maximum(-der2pp,FLOAT_EPS)
            der3ppp= -2*der1p*der2pp - der2pp*XB.reshape(N,T) - der1p
        if model == "logit" or model == "probit":
            der1b  = (der1p.reshape(-1,1)*X).reshape(N,T,dimX)
            der2pb = (der2pp.reshape(-1,1)*X).reshape(N,T,dimX)
            der3ppb= (der3ppp.reshape(-1,1)*X).reshape(N,T,dimX)
        if model =="ologit": 
            low = thresh[Y.astype('i')] - XB
            upp = thresh[Y.astype('i')+1] - XB
            cdf_low = getCdf(low)
            cdf_upp = getCdf(upp)
            pdf_low = cdf_low*(1-cdf_low)
            pdf_upp = cdf_upp*(1-cdf_upp)
            pdf_upplow = pdf_upp - pdf_low
            cdf_upplow = np.maximum(cdf_upp - cdf_low,FLOAT_EPS)
            derpdf_low = cdf_low-3*cdf_low**2+2*cdf_low**3
            derpdf_upp = cdf_upp-3*cdf_upp**2+2*cdf_upp**3
            derpdf_upplow = derpdf_upp - derpdf_low
            derderpdf_low = cdf_low-7*cdf_low**2+12*cdf_low**3-6*cdf_low**4
            derderpdf_upp = cdf_upp-7*cdf_upp**2+12*cdf_upp**3-6*cdf_upp**4
            derderpdf_upplow = derderpdf_upp - derderpdf_low
            
            der1p = -(pdf_upplow/cdf_upplow).reshape(N,T)
            der2pp = (derpdf_upplow/cdf_upplow - (pdf_upplow/cdf_upplow)**2).reshape(N,T)
            der3ppp = -(derderpdf_upplow/cdf_upplow- derpdf_upplow*pdf_upplow/(cdf_upplow**2) -2*(pdf_upplow/cdf_upplow*der2pp.reshape(-1,1))).reshape(N,T)
            
            der1b   = np.zeros((N*T,dimX+Ymax-1))
            der2pb  = np.zeros((N*T,dimX+Ymax-1))
            der3ppb = np.zeros((N*T,dimX+Ymax-1))
            der1b[:,:dimX]   = (der1p.reshape(-1,1)*X)
            der2pb[:,:dimX]  = (der2pp.reshape(-1,1)*X)
            der3ppb[:,:dimX] = (der3ppp.reshape(-1,1)*X)
            
            smallest1 =  (pdf_upp/cdf_upplow)
            largest1  = -(pdf_low/cdf_upplow)
            smallest2 =  -( derpdf_upp/cdf_upplow - pdf_upplow*  pdf_upp /(cdf_upplow**2))
            largest2  =  -(-derpdf_low/cdf_upplow - pdf_upplow*(-pdf_low)/(cdf_upplow**2))
            smallest3 = ( derderpdf_upp/cdf_upplow - derpdf_upp*pdf_upplow/(cdf_upplow**2) -( (derpdf_upplow*( pdf_upp)+pdf_upplow*( derpdf_upp))/(cdf_upplow**2)  - (pdf_upplow*( pdf_upp)*2*cdf_upplow*pdf_upplow)/(cdf_upplow**4) ))
            largest3  = (-derderpdf_low/cdf_upplow + derpdf_low*pdf_upplow/(cdf_upplow**2) -( (derpdf_upplow*(-pdf_low)+pdf_upplow*(-derpdf_low))/(cdf_upplow**2)  - (pdf_upplow*(-pdf_low)*2*cdf_upplow*pdf_upplow)/(cdf_upplow**4) ))
            for jj in range(1,Ymax):
                der1b[:,dimX+jj-1]  = (smallest1*(Y==jj)+(smallest1+largest1)*((Y>jj)&(Y<Ymax))+largest1*(Y==Ymax)).reshape(-1)
                der2pb[:,dimX+jj-1] = (smallest2*(Y==jj)+(smallest2+largest2)*((Y>jj)&(Y<Ymax))+largest2*(Y==Ymax)).reshape(-1)
                der3ppb[:,dimX+jj-1]= (smallest3*(Y==jj)+(smallest3+largest3)*((Y>jj)&(Y<Ymax))+largest3*(Y==Ymax)).reshape(-1)
            der1b[:,dimX:]  =   der1b[:,dimX:]*np.exp(paras[N+T-1+dimX:]).reshape(1,-1)
            der2pb[:,dimX:] =  der2pb[:,dimX:]*np.exp(paras[N+T-1+dimX:]).reshape(1,-1)
            der3ppb[:,dimX:]= der3ppb[:,dimX:]*np.exp(paras[N+T-1+dimX:]).reshape(1,-1)
            der1b  =  der1b.reshape(N,T,dimX+Ymax-1)
            der2pb = der2pb.reshape(N,T,dimX+Ymax-1)
            der3ppb=der3ppb.reshape(N,T,dimX+Ymax-1)
        return der1p,der1b,der2pp,der2pb,der3ppp,der3ppb
    # Helper function for computing the first and second order derivatives of APE wrt FE parass
    def panelApeHelper(paras):
        if model !="mlogit": thetas,XB = getLinprd(paras)
        if model =="ologit": 
            thresh = np.concatenate(([-np.inf], np.zeros(1)+cutoff0, paras[-(Ymax-1):].reshape(-1),[np.inf]))
        if model == "logit":  
            L = getCdf(XB) 
            first_der_lambda_it =  L-3*L**2+2*L**3
            secon_der_lambda_it =  L-7*L**2+12*L**3-6*L**4
        if model == "probit": 
            phi = getPdf(XB)
            first_der_lambda_it = -phi*XB 
            secon_der_lambda_it =  phi*XB**2-phi
        if model == "logit" or model == "probit":
            first_der_theta_it =  first_der_lambda_it*X
            first_der_theta  = (1/(Nold*Told))*(thetas@first_der_theta_it.sum(axis=0).reshape(1,-1)
                                          +np.diag(np.repeat(getPdf(XB).sum(),dimX)))
            first_der_lambda_mat =  (1/(Nold*Told))*(thetas.reshape(-1)*first_der_lambda_it).reshape(N,T,dimX)
            second_der_lambda_mat = (1/(Nold*Told))*(thetas.reshape(-1)*secon_der_lambda_it).reshape(N,T,dimX)
            for kx in range(dimX):
                if dummyIndicator[kx]==True: 
                    X0B = XB - X[:,kx].reshape(-1,1)*thetas[kx]
                    X1B = X0B + np.ones((N*T,1))*thetas[kx]
                    PDF1 = getPdf(X1B); PDF0 = getPdf(X0B)
                    CDF1 = getCdf(X1B); CDF0 = getCdf(X0B)
                    X1=np.copy(X); X0=np.copy(X)
                    X1[:,kx] = 1
                    X0[:,kx] = 0
                    first_der_theta[kx,:]  = (1/(Nold*Told))*(PDF1*X1 - PDF0*X0).sum(axis=0)
                    first_der_lambda_mat[:,:,kx] = (1/(Nold*Told))*(PDF1 - PDF0).reshape(N,T)
                    if model == "logit":  secon_der_lambda_it_kx = (CDF1-3*CDF1**2+2*CDF1**3)-(CDF0-3*CDF0**2+2*CDF0**3)
                    if model == "probit": secon_der_lambda_it_kx = -PDF1*X1B - (-PDF0*X0B)
                    second_der_lambda_mat[:,:,kx] = (1/(Nold*Told))*(secon_der_lambda_it_kx.reshape(N,T))
            first_der_lambda = np.vstack((first_der_lambda_mat.sum(axis=1),first_der_lambda_mat.sum(axis=0)))[1:,:]
            second_der_lambda_diag = np.vstack((second_der_lambda_mat.sum(axis=1),second_der_lambda_mat.sum(axis=0))) #[1:,:]
        if model =="ologit": 
            cdf    = getCdf(thresh-XB) 
            pdf    = cdf - cdf**2 
            derpdf    = cdf-3*cdf**2+2*cdf**3
            derderpdf = cdf-7*cdf**2+12*cdf**3-6*cdf**4
            firstder = np.zeros((dimX,Ymax+1,N+T-1+dimX+Ymax-1))
            second_der_lambda_diag = np.zeros((dimX,Ymax+1,N+T-1))
            for jj in range(Ymax+1):
                pdfdiff = (pdf[:,jj+1] - pdf[:,jj])
                derpdfdiff = (derpdf[:,jj+1] - derpdf[:,jj])
                firstder[:,jj,:N+T-1] = np.hstack(((-(derpdfdiff*thetas).reshape(dimX,N,T))[:,1:,:].sum(axis=2)/(N*T), -(derpdfdiff*thetas).reshape(dimX,N,T).sum(axis=1)/(N*T)))
                firstder[:,jj,N+T-1:N+T-1+dimX] = thetas@((derpdfdiff*(-X.T))).mean(axis=1).reshape(1,-1)+np.diag(np.repeat(pdfdiff.mean(),dimX))
                if jj ==1:
                    firstder[:,jj,N+T-1+dimX] = (derpdf[:,jj+1]*thetas).mean(axis=1)
                if jj >1 and jj<Ymax:
                    firstder[:,jj,N+T-1+dimX:] = np.hstack(((-derpdf[:,jj]*thetas).mean(axis=1).reshape(-1,1),(derpdf[:,jj+1]*thetas).mean(axis=1).reshape(-1,1)))
                if jj==Ymax:
                    firstder[:,jj,-1] = -(derpdf[:,jj]*thetas).mean(axis=1)
                derderpdfdiff = derderpdf[:,jj+1] - derderpdf[:,jj]
                tempvar = (derderpdfdiff*thetas)
                second_der_lambda_diag[:,jj,:] = np.hstack((tempvar.reshape(dimX,N,T)[:,1:,:].sum(axis=2),tempvar.reshape(dimX,N,T).sum(axis=1)))/(N*T)
            for kx in range(dimX):
                if dummyIndicator[kx]==True: 
                    X0B = XB - X[:,kx].reshape(-1,1)*thetas[kx]
                    X1B = X0B + np.ones((N*T,1))*thetas[kx]
                    CDF1 = getCdf(thresh-X1B); CDF0 = getCdf(thresh-X0B)
                    PDF1 = getPdf(thresh-X1B); PDF0 = getPdf(thresh-X0B)
                    derPDF1 = CDF1-3*CDF1**2+2*CDF1**3
                    derPDF0 = CDF0-3*CDF0**2+2*CDF0**3
                    X1=np.copy(X); X0=np.copy(X)
                    X1[:,kx] = 1
                    X0[:,kx] = 0
                    for jj in range(Ymax+1):
                        pdfdiff = (PDF1[:,jj+1] - PDF1[:,jj])-((PDF0[:,jj+1] - PDF0[:,jj]))
                        firstder[kx,jj,:N+T-1] = np.hstack((  (pdfdiff.reshape(N,T))[1:,:].sum(axis=1)/(N*T), (pdfdiff.reshape(N,T)).sum(axis=0)/(N*T))  )
                        firstder[kx,jj,N+T-1:N+T-1+dimX] = (((PDF1[:,jj+1] - PDF1[:,jj])*(X1.T))-((PDF0[:,jj+1] - PDF0[:,jj])*(X0.T))).mean(axis=1).reshape(1,-1)  
                        if jj ==1:
                            firstder[kx,jj,N+T-1+dimX] = -((PDF1-PDF0)[:,jj+1]).mean()
                        if jj >1 and jj<Ymax:
                            firstder[kx,jj,N+T-1+dimX:] = np.hstack((((PDF1-PDF0)[:,jj]).mean().reshape(-1,1),(-(PDF1-PDF0)[:,jj+1]).mean().reshape(-1,1)))
                        if jj==Ymax:
                            firstder[kx,jj,-1] = ((PDF1-PDF0)[:,jj]).mean()
                        derpdfdiff = (derPDF1[:,jj+1] - derPDF1[:,jj])-((derPDF0[:,jj+1] - derPDF0[:,jj]))
                        second_der_lambda_diag[kx,jj,:] = -np.hstack((derpdfdiff.reshape(N,T)[1:,:].sum(axis=1),derpdfdiff.reshape(N,T).sum(axis=0)))/(N*T)

            first_der_lambda = firstder[:,:,:N+T-1]
            first_der_theta  = firstder[:,:,N+T-1:]
            first_der_lambda_mat=0
        if model =="mlogit":
            global globalkk
            global globaljj
            apejacobia = np.zeros((dimX,Ymax,dimBigX*2))
            apehessian = np.zeros((dimX,Ymax,dimBigX*2,dimBigX*2))
            for kk in range(dimX):
                for jj in range(Ymax):
                    globalkk=kk
                    globaljj=jj
                    apejacobia[kk,jj,:  ]=torch.autograd.functional.jacobian(panelApeCalc_mlogit_tc, torch.from_numpy(paras).requires_grad_(True)).numpy()
                    apehessian[kk,jj,:,:]=torch.autograd.functional.hessian( panelApeCalc_mlogit_tc, torch.from_numpy(paras).requires_grad_(True)).numpy()
            first_der_theta  = apejacobia.reshape(dimX,Ymax, -1, Ymax, order='F')[:,:,-dimX:,:].reshape(dimX,Ymax, -1, order='F')
            first_der_lambda = apejacobia.reshape(dimX,Ymax, -1, Ymax, order='F')[:,:,:N+T-1,:].reshape(dimX,Ymax, -1, order='F')
            second_der_lambda_diag = apehessian.reshape(dimX,Ymax,dimBigX,Ymax,dimBigX,Ymax,order='F')[:,:,:N+T-1,:,:N+T-1,:].reshape(dimX,Ymax, (N+T-1)*Ymax, (N+T-1)*Ymax, order='F')
            first_der_lambda_mat=0
        return first_der_theta,first_der_lambda,second_der_lambda_diag,first_der_lambda_mat
    #--------------------------------------------------------------------------------
    # Functions for further analytical bias corrections for common parameter and APE 
    #--------------------------------------------------------------------------------
    # Further analytical bias corrections for APE after likelihood correction
    def panelApeBiasTerm(paras):
        _, _, second_der_lambda_diag,_ = panelApeHelper(paras)
        if model == "logit" or model == "probit":
            second_der_lambda_diag=(second_der_lambda_diag*(Nold*Told)).reshape(-1,dimX)
            _,XB = getLinprd(paras)
        if model == "logit":  
            L = getCdf(XB) 
            der2pp = -(L-L**2).reshape(N,T)
        if model == "probit":
            pdf = getPdf(XB).reshape(N,T)
            cdf = getCdf(XB).reshape(N,T)
            der2pp = -(pdf**2) / (cdf*getCdf(-XB).reshape(N,T))
        if model == "logit" or model == "probit":
            ape_bias1 = -(second_der_lambda_diag[:N,:]/(der2pp.sum(1).reshape(-1,1))).mean(0)/T
            ape_bias0 = -(second_der_lambda_diag[N:,:]/(der2pp.sum(0).reshape(-1,1))).mean(0)/N
            ape_bias = ape_bias1+ape_bias0
        if model =="ologit": 
            invH_diag = 1/(np.diag( panelNegHess(paras) )[:N+T-1])
            ape_bias = (second_der_lambda_diag[:,]*invH_diag).sum(axis=2)
        if model =="mlogit": 
            HHH = panelNegHess(paras).reshape(dimBigX,Ymax,dimBigX,Ymax,order='F')[:N+T-1,:,:N+T-1,:].reshape((N+T-1)*Ymax, (N+T-1)*Ymax, order='F')
            HHH = HHH*np.kron(np.eye(N+T-1),np.ones((Ymax,Ymax)))
            invH = np.linalg.inv(HHH)
            second_der_lambda_diag = second_der_lambda_diag*(Nold*Told)*(np.kron(np.eye(N+T-1),np.ones((Ymax,Ymax))).reshape(1,1,(N+T-1)*Ymax,(N+T-1)*Ymax))
            ape_bias = np.diagonal((second_der_lambda_diag@invH),axis1=2,axis2=3).sum(2)/(N*T) #*0
        return ape_bias/2 
    # Further analytical bias corrections for common parameter and APE (if StaticExp and ebc are claimed) (fast projection version)
      # -- binary response model only
    def panelStaticExpBiasTerm(paras):
        _,XB = getLinprd(paras)
        _,_,_,date= panelApeHelper(paras)
        date=date*(Nold*Told)
        if model == "logit":  
            L = getCdf(XB) 
            psi  = (Y-L).reshape(N,T)
            ws = (L-L**2).reshape(N,T)
        if model == "probit":
            pdf = getPdf(XB).reshape(N,T)
            cdf = getCdf(XB).reshape(N,T)
            ws = (pdf**2) / (cdf*getCdf(-XB).reshape(N,T))
            psi  = (ws*(Y.reshape(N,T)-cdf))/ pdf
        fenmu1 = ws.sum(1)
        fenmu0 = ws.sum(0)
        numpara=dimX
        sP2bL=np.zeros(numpara); sP3bL=np.zeros(numpara)
        WWW = np.diag(ws.reshape(-1))
        proj = np.linalg.inv(FE.T@WWW@FE)@FE.T@WWW
        spans0 = proj@X.reshape(-1,numpara)
        YYY2 = ((date.reshape(-1,numpara))/ws.reshape(-1,1))
        spans2 = proj@YYY2
        resx = (X-(FE@spans0))
        rdate = (YYY2 - FE@spans2)
        maJiang0 =  resx.reshape(N,T,numpara) * ws.reshape(N,T,1)
        maJiang2 = rdate.reshape(N,T,numpara) * ws.reshape(N,T,1)
        if priorLag >0:
            for ll in range(1,priorLag+1):
                sP2bL = sP2bL -( ( maJiang0[:,ll: ,:]*((psi.reshape(N,T,1))[:,:-ll,:]) ).sum(1) / fenmu1.reshape(-1,1) ).mean(0)/(T-ll)
                sP3bL = sP3bL +( ( maJiang2[:,ll: ,:]*((psi.reshape(N,T,1))[:,:-ll,:]) ).sum(1) / fenmu1.reshape(-1,1) ).mean(0)/(T-ll)
        if priorVersion is not None:  
            sP2b= (   ( (maJiang0*psi.reshape(N,T,1)).sum(1) / fenmu1.reshape(-1,1) ).mean(0)/T
                    + ( (maJiang0*psi.reshape(N,T,1)).sum(0) / fenmu0.reshape(-1,1) ).mean(0)/N
                  ) +sP2bL
            sP3b= (   ( (maJiang2*psi.reshape(N,T,1)).sum(1) / fenmu1.reshape(-1,1) ).mean(0)/T
                    + ( (maJiang2*psi.reshape(N,T,1)).sum(0) / fenmu0.reshape(-1,1) ).mean(0)/N
                  ) +sP3bL
        if priorVersion is None:
            sP2b = sP2bL
            sP3b = sP3bL
            if model == "logit":
                sP2b = sP2b + (( (maJiang0*getCdf(XB).reshape(N,T,1)).sum(1) / fenmu1.reshape(-1,1) ).mean(0)/T
                             + ( (maJiang0*getCdf(XB).reshape(N,T,1)).sum(0) / fenmu0.reshape(-1,1) ).mean(0)/N )
                pdate = (FE@spans2).reshape(N,T,numpara)
                Edate = ((L-3*L**2+2*L**3)).reshape(N,T,1)
                sP3b = sP3b + 0.5*(((-Edate*pdate).sum(1) / fenmu1.reshape(-1,1) ).mean(0)/T
                                 + ((-Edate*pdate).sum(0) / fenmu0.reshape(-1,1) ).mean(0)/N ) 
            if model == "probit":
                sP2b = sP2b + 0.5*(( (maJiang0*(XB).reshape(N,T,1)).sum(1) / fenmu1.reshape(-1,1) ).mean(0)/T
                                 + ( (maJiang0*(XB).reshape(N,T,1)).sum(0) / fenmu0.reshape(-1,1) ).mean(0)/N )
                pindex = (FE@(proj@XB.reshape(-1,1))).reshape(N,T,1)
                sP3b = sP3b + 0.5*(( (date*pindex).sum(1) / fenmu1.reshape(-1,1) ).mean(0)/T
                                 + ( (date*pindex).sum(0) / fenmu0.reshape(-1,1) ).mean(0)/N ) 
        # Point estimate - common parameters - Further Bias Term
        W = (resx.T @ (ws.reshape(-1,1)*resx))/(N*T)
        biasTheta = np.linalg.inv(W)@( sP2b.reshape(-1,1) )
        #invW = np.linalg.inv( panelNegHess(paras))[N+T-1:,N+T-1:]
        #biasTheta = invW*(N*T)@(-sP2b.reshape(-1,1) )
        # Point estimate - APE parameters - Further Bias Term
        biasApe = sP3b 
        return biasTheta.reshape(-1),biasApe
    #--------------------------------------------------------------------------------
    # Torch functions for MNLogit 
    #--------------------------------------------------------------------------------
    if model=='mlogit': 
        bigX_tc        = torch.from_numpy(bigX   ) 
        Ydummies_tc    = torch.from_numpy(Ydummies   ) 
    def reshape_fortran(x, shape):
        if len(x.shape) > 0:
            x = x.permute(*reversed(range(len(x.shape))))
        return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))
    def getPdf_tc(arg):
        return torch.column_stack((torch.ones(len(arg)), torch.exp(arg)))
    def getCdf_tc(arg):
        eXB = getPdf_tc(arg)
        return eXB/eXB.sum(1)[:,None]
    def panelApeCalc_mlogit_tc(paras):
        global globalkk
        global globaljj
        if model == "mlogit": 
            thetas = (reshape_fortran(paras,(dimBigX, -1)))[-dimX:,:]
            pr = getCdf_tc((bigX_tc@reshape_fortran(paras,(dimBigX, -1))))
            L = pr[:,1:]
            bigL = (thetas[:,None]*L).permute(0,2,1)
            bigLSumj = bigL.sum(1)
            ape_mat = torch.zeros(( dimX, Ymax, N, T ))
            for kk in range(dimX):
                if dummyIndicator[kk]==True:
                    bigX0=torch.clone(bigX_tc)
                    bigX1=torch.clone(bigX_tc)
                    bigX0[:,N+T-1+kk]=0
                    bigX1[:,N+T-1+kk]=1
                    L0 = (getCdf_tc((bigX0@reshape_fortran(paras,(dimBigX, -1)))))[:,1:]
                    L1 = (getCdf_tc((bigX1@reshape_fortran(paras,(dimBigX, -1)))))[:,1:]
                    ape_mat[kk,:,:,:] = (L1-L0).reshape(Ymax,N,T)
                else:
                    ape_mat[kk,:,:,:] = (bigL[kk,:,:] - bigLSumj[kk,:]*L.T).reshape(Ymax,N,T)
            ape = (ape_mat.sum(2).sum(2))/(Nold*Told)
        return ape[globalkk,globaljj]
    def panelNegLogLike_mlogit_tc(paras):
        pr = getCdf_tc(bigX_tc@reshape_fortran(paras,(dimBigX, -1)))
        L = pr[:,1:]
        logLike = torch.sum(Ydummies_tc * torch.log(pr))
        # --- Compute hessian ---
        if priorVersion is not None:
            partials=[]
            for i in range(Ymax):
                for j in range(Ymax): 
                    if i == j:
                        partials.append(-(pr[:,i+1]*(1-pr[:,j+1])).reshape(N,T))
                    else:
                        partials.append(-(pr[:,i+1]*( -pr[:,j+1])).reshape(N,T))
            der2pp = (torch.stack(partials, dim=0)).permute(1,2,0)
            diagNegH =  torch.cat(( torch.linalg.det(-der2pp.sum(1).reshape(N,Ymax,Ymax))[1:],
                                    torch.linalg.det(-der2pp.sum(0).reshape(T,Ymax,Ymax))  ))
        # --- Compute HAC Score Outer Prod ---
            if priorVersion is not None and priorVersion !="StaticExp":
                mat=torch.linalg.inv(torch.cat((-der2pp.sum(1)[1:,:].reshape(N-1,Ymax,Ymax),-der2pp.sum(0).reshape(T,Ymax,Ymax))))
                invdiagNegH = torch.block_diag(*mat)
                # --- Compute HAC Score Outer Prod (Lag0) ---
                #diagSS  = torch.append((der1p**2).sum(1)[1:],(der1p**2).sum(0))
                if model =="mlogit": 
                    der1p = (Ydummies_tc[:,1:]*1 - L ).reshape(N,T,Ymax)
                    temps = torch.tensordot(der1p[:,:,None].permute(0,1,3,2),der1p[:,:,None], dims=([3],[2]))
                    term1 = torch.diagonal(torch.diagonal(temps,dim1=0,dim2=3),dim1=0,dim2=2).sum(3).permute(2,0,1)
                    term2 = torch.diagonal(torch.diagonal(temps,dim1=1,dim2=4),dim1=0,dim2=2).sum(3).permute(2,0,1)
                    diagSS = torch.block_diag(*(torch.cat((term1[1:,:,:],term2))))
                # --- Compute HAC Score Outer Prod (Lag1) ---
                if priorLag >= 1:
                    if model =="mlogit": 
                        temps = torch.tensordot(der1p[:,:-1,None].permute(0,1,3,2),der1p[:,1:,None], dims=([3],[2]))
                        term1 = torch.diagonal(torch.diagonal(temps,dim1=0,dim2=3),dim1=0,dim2=2).sum(3).permute(2,0,1)
                        diagSSLag = 2*T/(T-1)*torch.block_diag(*(torch.cat((term1[1:,:,:],torch.zeros((T,Ymax,Ymax))))))
                # --- Compute HAC Score Outer Prod (Lag2+) ---
                if priorLag > 1:
                    for ll in range(2,priorLag+1):
                        if model =="mlogit": 
                            #diagSSLag = diagSSLag+ torch.cat((2*T/(T-ll)*(  (der1p[:,:-ll,:]*der1p[:,ll:])[1:,:,:]  ).sum(axis=1),torch.zeros((T,Ymax)))).reshape(-1)
                            temps = torch.tensordot(der1p[:,:-ll,None].permute(0,1,3,2),der1p[:,ll:,None], dims=([3],[2]))
                            term1 = torch.diagonal(torch.diagonal(temps,dim1=0,dim2=3),dim1=0,dim2=2).sum(3).permute(2,0,1)
                            diagSSLag = diagSSLag+ 2*T/(T-ll)*torch.block_diag(*(torch.cat((term1[1:,:,:],torch.zeros((T,Ymax,Ymax))))))
        # --- Different Versions of Priors ---
        if priorVersion is None: prior = 0
        if priorVersion =="StaticExp": 
            prior = 0.5*torch.log(diagNegH).sum()
        if priorVersion == "Generic" and priorLag==0:
            if model !="mlogit": prior = -0.5*((diagSS)/diagNegH).sum()
            if model =="mlogit": prior = -0.5*(torch.diag(invdiagNegH@diagSS).sum())
        if priorVersion == "Generic" and priorLag>=1:
            if model !="mlogit": prior = -0.5*((diagSSLag+diagSS)/diagNegH).sum()
            if model =="mlogit": prior = -0.5*(torch.diag(invdiagNegH@(diagSSLag+diagSS)).sum())  
        if priorVersion == "Binary" and priorLag==0:
            prior = 0.5*torch.log(diagNegH).sum()
        if priorVersion == "Binary" and priorLag>=1:
            if model !="mlogit": prior = -0.5*((diagSSLag)/diagNegH)[:N-1].sum() + 0.5*(torch.log(diagNegH).sum())
            if model =="mlogit": prior = -0.5*(torch.diag(invdiagNegH@(diagSSLag)).sum())  + 0.5*(torch.log(diagNegH).sum())
        return -(logLike + prior )
    #----------------------------------------------------------
    # Implementations
    #----------------------------------------------------------
    start_time = time.time()  # Setup a timer
    # [> Point estimation - No bias correction or prior for the full likelihood <]
    # JML
    algorithmraw=algorithm+""
    if algorithm!="JML": algorithm="JML"
    meth_use = 'BFGS' # MLE method can be BFGS or Newton-CG, for MLE with prior, hessian of prior is not provided, and hence BFGS can be more efficient
    if model !="mlogit": res = sp.optimize.minimize(panelNegLogLike,x0=sv, method=meth_use, jac=panelNegScore)
    if model =="mlogit" and priorVersion is None: res = sp.optimize.minimize(panelNegLogLike,x0=sv.reshape(-1), method=meth_use, jac=panelNegScore)
    if model =="mlogit" and priorVersion is not None: 
        #res = sp.optimize.minimize(panelNegLogLike,x0=sv.reshape(-1), method=meth_use)
        res = minimize2(panelNegLogLike_mlogit_tc,x0=torch.from_numpy(sv.reshape(-1)),method=meth_use)
        res.fun=res.fun.numpy()
        res.x=res.x.numpy()
    sucMess = res.success
    funcLogLike = res.fun
    estAll   = res.x.reshape(-1)
    if algorithmraw!="JML": algorithm = algorithmraw
    # Iterative MLE algorithm - No bias correction or prior for the full likelihood
    if algorithm == "Iter":
        # Starting values -- from JML to speed up the algorithm
            ## ( here, parameterUpdate will be a global variable in this algorithm)
        parameterUpdate = np.copy(estAll) # parameterUpdate will be a global variable in this algorithm
        # Initilizations
        funVal = -999999
        np.random.seed(1)
        iterlist = range(2000)
        # Iterative MLE starts here
        for sweep in iterlist:
            np.random.seed(sweep)
            stepsize = np.random.randint(50,90)
            newindices=np.arange(np.size(sv))
            np.random.shuffle(newindices)
            for start_i in np.arange(0,np.size(sv),stepsize):
                end_i = start_i+stepsize
                parameterSlice = parameterUpdate[newindices[start_i:end_i]]
                if model !="mlogit": res = sp.optimize.minimize(panelNegLogLikeConcen,x0=parameterSlice, method=meth_use, jac=panelNegScoreConcen)
                if model =="mlogit": res = sp.optimize.minimize(panelNegLogLikeConcen,x0=parameterSlice, method=meth_use)
                if start_i==0: 
                    funValDiff = np.abs(funVal - res.fun)
                    funVal = res.fun
            # Exit criteron 
                ## ( this is a bit strict -- to ensure more precise estimates obtained
                ##   by scipy optimization toolbox)
            if funValDiff<1e-15:
                sucMess = res.success
                funcLogLike = res.fun
                break
        estAll = np.copy(parameterUpdate)
    # MCMC algorithm - Prior for the integrated likelihood
    if algorithm == "MCMC":
        # Starting values -- by default, it starts from JML to speed up the algorithm
        #                 -- otherwise, use the starting values of all zeroes for FE parameters and 0.5 for beta parameters by setting `sv_mle=False`
        sucMess=1
        estAllcopy = np.copy(estAll)
        if model=='ologit': # For ologit, after reparametrization, the cutoff points should be recovered 
            estAllcopy[-(Ymax-1):] = cutoffTrans(estAllcopy[-(Ymax-1):], cutoff0)
        if mcmc_sv_mle==True:
            allpara   = np.copy(estAll)
        else:
            allpara   = np.copy(sv)
        allpara_star = np.copy(allpara)
        if model == 'ologit': allparaTrans=np.copy(allpara)
        # Empty matrices for save the chain
        if model != "mlogit": lambda_mcmc=np.zeros((mcmc_iters,N+T-1))
        if model == "logit" or model == "probit": 
            theta_mcmc =np.zeros((mcmc_iters,dimX))
            #ape_mcmc   =np.zeros((mcmc_iters,dimX))
            allparaNum = N+T-1+dimX
        if model == "ologit": 
            theta_mcmc =np.zeros((mcmc_iters, dimX+Ymax-1))
            #ape_mcmc   =np.zeros((mcmc_iters,  dimX,  Ymax+1))
            allparaNum = N+T-1+dimX+Ymax-1
        if model == "mlogit": 
            lambda_mcmc=np.zeros((mcmc_iters,(N+T-1)*Ymax))
            theta_mcmc =np.zeros((mcmc_iters, dimX*Ymax))
            allparaNum = dimBigX*Ymax
        varchooseScale = (2.38**2)/allparaNum
        # MCMC starts
        mcmcIterList=(range(mcmc_iters))
        newindices=np.arange(allparaNum) 
        if mcmc_timer==True: mcmcIterList=tqdm(range(mcmc_iters))
        for m in mcmcIterList:    
            # Fix the PRNG seed
            np.random.seed(m)
            # Step 1 Propose new values
            allpara_new = allpara + np.random.normal(loc=0,scale=para_sd)
            if m>=400:
                varchoose1 = varchooseScale * np.cov(np.hstack((lambda_mcmc,theta_mcmc))[50:m-1,:].T) 
                try: 
                    allpara_new = 0.1*allpara_new+0.9*allpara+ np.linalg.cholesky(varchoose1)@np.random.standard_normal(size=(allparaNum))
                except:
                    allpara_new = 0.1*allpara_new+0.9*allpara+ np.random.multivariate_normal(mean=np.zeros(allparaNum),cov=varchoose1, check_valid='ignore')
            # Step 2: Update Decision 
            ## Random blocking setup
            np.random.shuffle(newindices)
            ## MH starts
            update = 1
            for start_i in np.arange(0,allparaNum,block_size):
                end_i = start_i+block_size
                allpara_star[:] = allpara[:]
                allpara_star[newindices[start_i:end_i]] = allpara_new[newindices[start_i:end_i]]
                if update == 1: posteriorOld= -panelNegLogLike(allpara)
                posteriorNew= -panelNegLogLike(allpara_star)
                if posteriorNew-posteriorOld>=np.log(np.random.sample() ):  
                    allpara[:] = allpara_star[:]
                    update = 1
                else:
                    update = 0 
            # Update parameters
            if model != "mlogit":  
                theta_mcmc[m,:] =allpara[N+T-1:]
                lambda_mcmc[m,:]=allpara[:N+T-1]  
            if model == "mlogit":  
                theta_mcmc[m,:] =allpara[(N+T-1)*Ymax:]
                lambda_mcmc[m,:]=allpara[:(N+T-1)*Ymax]  
            # APE as a function of posterior
            #if ape_compute==True: 
            #    if model == "logit" or model == "probit": 
            #        ape_mcmc[m,:] = (panelApeCalc(allpara))[0]
            #    if model == "ologit": 
            #        allparaTrans[:] = allpara[:]
            #        allparaTrans[-(Ymax-1):] = cutoffTrans(allparaTrans[-(Ymax-1):], cutoff0)
            #        ape_mcmc[m,:,:] = (panelApeCalc(allparaTrans))[0]
    # [> Organize parameters to output <]   
    if algorithm=="MCMC": 
        if model=='ologit': theta_mcmc[:,dimX:] = (np.hstack((np.zeros((mcmc_iters,1))+cutoff0,np.exp(theta_mcmc[:,dimX:]))).cumsum(axis=1))[:,1:]
        lambda_mcmc_burnin = lambda_mcmc[mcmc_burnin:,:]
        theta_mcmc_burnin  = theta_mcmc[mcmc_burnin:,:]
        estThetaSD = theta_mcmc_burnin[0::mcmc_skipsize,:].std(0)
        estAll=np.append(lambda_mcmc_burnin[0::mcmc_skipsize,:].mean(0),theta_mcmc_burnin[0::mcmc_skipsize,:].mean(0))
        accRateTheta = np.diff(theta_mcmc_burnin,axis=0)
        accRateTheta[accRateTheta!=0] = 1
        accRateTheta = accRateTheta.mean(0)*100
        accRateFE    = np.diff(lambda_mcmc_burnin,axis=0)
        accRateFE[accRateFE!=0] = 1
        accRateFE = accRateFE.mean(0)*100
    if algorithm!="MCMC": 
        if model=='ologit': # For ologit, after reparametrization, the cutoff points should be recovered 
            estAll[-(Ymax-1):] = cutoffTrans(estAll[-(Ymax-1):], cutoff0)
    if model != "mlogit":  
        estFE = estAll[:N+T-1]
        estTheta = estAll[N+T-1:]
    # [> Asymptotic standard errors <]
    if model != "mlogit":  
        model_paraSE = np.sqrt(np.diag(np.linalg.inv(panelNegHess(estAll)))[N+T-1:]) 
    if model == "mlogit":  
        model_paraSE = np.sqrt(np.diag(np.linalg.inv(panelNegHess(estAll))).reshape(-1,Ymax, order='F'))[N+T-1:,:]
    if ape_compute==True: apese=panelApeSE(estAll) 
    # [> Analytical bias correction if `ebc==True` is specified <]
    if ebc is True:
        algorithmraw=algorithm
        algorithm="JML"
        furBias ,_  = panelStaticExpBiasTerm(estAll)
        estTheta = estTheta.reshape(-1)-furBias.reshape(-1)
        estAll[N+T-1:] = estTheta
        estAll[:N+T-1] = sp.optimize.minimize(panelNegLogLike_fixTheta,x0=estAll[:N+T-1], method='BFGS', jac=panelNegScore_fixTheta).x
    # [> APE <]
    if ape_compute==True:
        apeest=(panelApeCalc(estAll))[0]
        if priorVersion is not None or ebc is True:
            apeest=apeest - panelApeBiasTerm(estAll)
        # Analytical bias correction if `ebc==True` is specified
        if ebc is True:
            _ ,furApeBias  = panelStaticExpBiasTerm(estAll)
            apeest=apeest - furApeBias
    else:
        apeest=np.nan; apese=np.nan; apebc=np.nan
    # [> Housekeeping <]
    if algorithmraw!="MCMC": 
        theta_mcmc=np.nan; lambda_mcmc=np.nan
        apeest2=np.nan;apeest3=np.nan;apeSD2=np.nan;ape_mcmc=np.nan ; apeSD=np.nan; estThetaSD=np.nan; accRateTheta=np.nan; accRateFE=np.nan
    else:
        ape_mcmc=np.nan
        if model != "mlogit":
            theta_mcmc=theta_mcmc_burnin[0::mcmc_skipsize,:]; lambda_mcmc=lambda_mcmc_burnin[0::mcmc_skipsize,:]
        if model == "mlogit":
            all_mcmc_burnin = np.hstack((lambda_mcmc_burnin,theta_mcmc_burnin)).reshape(-1,dimBigX,Ymax,order='F')
            accRateTheta = np.diff(all_mcmc_burnin[:,(N+T-1):,:],axis=0)
            accRateFE    = np.diff(all_mcmc_burnin[:,:(N+T-1),:],axis=0)
            accRateTheta[accRateTheta!=0] = 1
            accRateTheta = accRateTheta.mean(0)*100
            accRateFE[accRateFE!=0] = 1
            accRateFE = accRateFE.mean(0)*100
            all_mcmc = np.hstack((lambda_mcmc_burnin[0::mcmc_skipsize,:],theta_mcmc_burnin[0::mcmc_skipsize,:])).reshape(-1,dimBigX,Ymax,order='F')
            theta_mcmc=all_mcmc[:,(N+T-1):,:]; lambda_mcmc=all_mcmc[:,:(N+T-1),:]
    if model == "mlogit":  
        estAll=estAll.reshape(-1,Ymax, order='F')
        estTheta=estAll[N+T-1:,:]
        estFE=estAll[:N+T-1,:]
        if algorithmraw=="MCMC": 
            estThetaSDall = np.append(lambda_mcmc_burnin[0::mcmc_skipsize,:].std(0),theta_mcmc_burnin[0::mcmc_skipsize,:].std(0))
            estThetaSD = (estThetaSDall.reshape(-1,Ymax, order='F'))[N+T-1:,:]
    end_time = time.time()
    return estTheta, model_paraSE, estFE, sucMess, -funcLogLike, end_time - start_time, apeest, apese, theta_mcmc, lambda_mcmc, estThetaSD,accRateTheta ,accRateFE
#not oOo