"""
Python functions for data generating process
of various network formation models

By Zizhong Yan (helloyzz@gmail.com)

Created on Tue Aug 26 2025
"""
#----------------------------------------------------------
# Load library dependencies
#----------------------------------------------------------
import numpy as np
import scipy as sp
from tqdm import tqdm 
from scipy import stats
from scipy.stats import norm,logistic
#------------------------------------------------------------------------
# Nonlinear panel model DGP: 
# ---  based on Fernanzed-Val and Weidner (2016) and has been extended to
#      binary probit/logit, ordered probit/logit, multinomial logit
#------------------------------------------------------------------------
def PanelGenData(N,T,seed="2025",model="logit",dynamic=0): 
    np.random.seed(seed)
    #index
    index_i  = np.repeat(np.arange(N),T)
    index_t  = np.tile(np.arange(T),N)
    #FE
    alphas0 = np.random.normal(loc=0  ,scale=0.25,size=(N,1))
    gammas0 = np.random.normal(loc=0  ,scale=0.25,size=(T,1)) 
    #FE added as dummy regressors 
    FE=np.zeros((N*T,N+T))
    column=0
    for i in np.unique(index_i):
        FE[index_i==i,column]=1
        column=column+1
    for t in np.unique(index_t):
        FE[index_t==t,column]=1
        column=column+1
    FE=np.delete(FE, 0, axis=1)
    FEi=FE[:,0:N-1]
    FEt=FE[:,N-1:N+T]
    if dynamic == 0:
        #X
        Xmat= np.zeros((N,T)) 
        X0=norm.rvs(loc=0, scale=1,size=N)
        for t in range(T):
            if t==0: Xmat[:,t]=X0*0.5 + alphas0.reshape(-1)+gammas0[t]+norm.rvs(loc=0, scale=np.sqrt(1/2),size=N)
            if t>0:  Xmat[:,t]=Xmat[:,t-1]*0.5 + alphas0.reshape(-1)+gammas0[t]+norm.rvs(loc=0, scale=np.sqrt(1/2),size=N)
        X = Xmat.reshape(N*T,1)
        #Error
        if model=="logit" or model=="ologit":
            Error = logistic.rvs( loc=0, scale=1, size=(N*T,1) )
        elif model=="probit" or model=="oprobit":
            Error = norm.rvs( loc=0, scale=1, size=(N*T,1) )
        elif model=="mlogit" :
            J=3
            Error = np.random.gumbel(size=(N*T,J) )
        #Y
        if model!='mlogit':
            Ystar = X + np.matmul(FEi,alphas0[1:])+np.matmul(FEt,gammas0) + Error
            Y = np.copy(Ystar)
            Y[Ystar >= 0] = 1
            Y[Ystar <  0] = 0        
        if model=='mlogit':
            Y=np.zeros((N*T,1))
            Y[((X.reshape(-1)*(-1)+Error[:,0]-Error[:,1])>0) * ((X.reshape(-1)*(-1)+Error[:,0]-Error[:,2])>0)]=0
            Y[((X.reshape(-1)*( 1)+Error[:,1]-Error[:,0])>0) * ((X.reshape(-1)*( 0)+Error[:,1]-Error[:,2])>0)]=1
            Y[((X.reshape(-1)*( 1)+Error[:,2]-Error[:,0])>0) * ((X.reshape(-1)*( 0)+Error[:,2]-Error[:,1])>0)]=2
            Ystar = np.copy(Y)
    if dynamic == 1:
        #Z
        Zmat= np.zeros((N,T)) 
        Z0=norm.rvs(loc=0, scale=1,size=N)
        for t in range(T):
            if t==0: Zmat[:,t]=Z0*0.5 + alphas0.reshape(-1)+gammas0[t]+norm.rvs(loc=0, scale=np.sqrt(1/2),size=N)
            if t>0:  Zmat[:,t]=Zmat[:,t-1]*0.5 + alphas0.reshape(-1)+gammas0[t]+norm.rvs(loc=0, scale=np.sqrt(1/2),size=N)
        #Lagged Y
        if model=="logit" or model=="ologit":
            Error0 = logistic.rvs( loc=0, scale=1, size=N )
        elif model=="probit" or model=="oprobit":
            Error0 = norm.rvs( loc=0, scale=1, size=N )
        elif model=="mlogit" :
            J=3
            Error0 = np.random.gumbel(size=(N,J) )
        if model!='mlogit':
            Y0star = Z0 + alphas0.reshape(-1)+np.random.normal(loc=0,scale=0.25) - Error0
            Y0 = np.copy(Y0star)
            Y0[Y0star >= 0] = 1
            Y0[Y0star < 0] = 0
            Ylag_mat = np.zeros((N,T))
            Ylag_mat[:,0] = Y0
        if model=='mlogit':
            Y0=np.zeros((N,1))
            Y0[((Z0*(-1)+Error0[:,0]-Error0[:,1])>0) * ((Z0*(-1)+Error0[:,0]-Error0[:,2])>0)]=0
            Y0[((Z0*( 1)+Error0[:,1]-Error0[:,0])>0) * ((Z0*( 0)+Error0[:,1]-Error0[:,2])>0)]=1
            Y0[((Z0*( 1)+Error0[:,2]-Error0[:,0])>0) * ((Z0*( 0)+Error0[:,2]-Error0[:,1])>0)]=2
            Ylag_mat = np.zeros((N,T))
            Ylag_mat[:,0] = (Y0==1).reshape(-1)*1
        Y_mat = np.zeros((N,T))
        Ystar_mat = np.zeros((N,T))
        #Y
        for t in range(T):
            if model=="logit" or model=="ologit":
                Error0 = logistic.rvs( loc=0, scale=1, size=N )
            elif model=="probit" or model=="oprobit":
                Error0 = norm.rvs( loc=0, scale=1, size=N )
            elif model=="mlogit" :
                J=3
                Error0 = np.random.gumbel(size=(N,J) )
            if model!='mlogit':
                if t==0: Ystar = Ylag_mat[:,0]*0.5+Zmat[:,t].reshape(-1) + alphas0.reshape(-1)+gammas0[t] - Error0
                if t>0:  Ystar = Y_mat[:,t-1] *0.5+Zmat[:,t].reshape(-1) + alphas0.reshape(-1)+gammas0[t] - Error0
                Y = np.copy(Ystar)
                Y[Ystar >= 0] = 1
                Y[Ystar < 0 ] = 0
                Y_mat[:,t] = Y
                Ystar_mat[:,t] = Ystar
                if t!=T-1: Ylag_mat[:,t+1] = Y
            if model=='mlogit':
                if t==0: 
                    Ystar=np.zeros((N))
                    Ystar[((Ylag_mat[:,0]*(-0.5)+Zmat[:,t]*(-1)+Error0[:,0]-Error0[:,1])>0) * ((Ylag_mat[:,0]*(-0.5)+Zmat[:,t]*(-1)+Error0[:,0]-Error0[:,2])>0)]=0
                    Ystar[((Ylag_mat[:,0]*( 0.5)+Zmat[:,t]*( 1)+Error0[:,1]-Error0[:,0])>0) * ((Ylag_mat[:,0]*(   0)+Zmat[:,t]*( 0)+Error0[:,1]-Error0[:,2])>0)]=1
                    Ystar[((Ylag_mat[:,0]*( 0.5)+Zmat[:,t]*( 1)+Error0[:,2]-Error0[:,0])>0) * ((Ylag_mat[:,0]*(   0)+Zmat[:,t]*( 0)+Error0[:,2]-Error0[:,1])>0)]=2
                if t>0:  
                    Ystar[((Ylag_mat[:,t]*(-0.5)+Zmat[:,t]*(-1)+Error0[:,0]-Error0[:,1])>0) * ((Ylag_mat[:,t]*(-0.5)+Zmat[:,t]*(-1)+Error0[:,0]-Error0[:,2])>0)]=0
                    Ystar[((Ylag_mat[:,t]*( 0.5)+Zmat[:,t]*( 1)+Error0[:,1]-Error0[:,0])>0) * ((Ylag_mat[:,t]*(   0)+Zmat[:,t]*( 0)+Error0[:,1]-Error0[:,2])>0)]=1
                    Ystar[((Ylag_mat[:,t]*( 0.5)+Zmat[:,t]*( 1)+Error0[:,2]-Error0[:,0])>0) * ((Ylag_mat[:,t]*(   0)+Zmat[:,t]*( 0)+Error0[:,2]-Error0[:,1])>0)]=2
                Y_mat[:,t] = Ystar
                Ystar_mat[:,t] = Ystar
                if t!=T-1: Ylag_mat[:,t+1] = (Ystar==1).reshape(-1)*1
        Y = Y_mat.reshape(-1,1)
        Ystar = Ystar_mat.reshape(-1,1)
        #X
        X = np.hstack((Ylag_mat.reshape(-1,1),Zmat.reshape(-1,1)))
    if model=="ologit" or model=="oprobit": Y[Ystar <  -2.5] = 0
    if model=="ologit" or model=="oprobit": Y[Ystar >= -2.5] = 1
    if model=="ologit" or model=="oprobit": Y[Ystar >=  0.5] = 2
    if model=="ologit" or model=="oprobit": Y[Ystar >=  2.5] = 3
    return FEi,FEt,FE,index_i,index_t,Y,X,Ystar,alphas0,gammas0


