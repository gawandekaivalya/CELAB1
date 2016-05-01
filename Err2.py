# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 20:34:11 2016

@author: Kaivalya
"""

import numpy as np
import scipy
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import statsmodels.stats.stattools as stools
import pandas as pd
from scipy.misc import derivative
import uncertainties
from uncertainties import unumpy
from uncertainties import ufloat

data=pd.read_excel("Data.xlsx","Sheet1")
Q=scipy.array(data["flowrate"])
delH=scipy.array(data["deltaH"])

L=2.69      #m
d=0.0198    #m
mu=0.00095  #Pa.s
Dw=1000     #kg/m^3
Dc=1487     #kg/m^3
g=9.81      #m/s^2
dQ=10**-6   #m^3/s
dH=0.001    #m
#print Q
#print delH

def partial_derivative(func, var=0, point=[]):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return derivative(wraps, point[var], dx = 1e-6)
'''    
def reynolds(x):
    return ((4*Dw*x)/(3.14*d*mu))

def fricfact(x,y):
    return ((y*9.81*(d**3)*(3.14**2)/(32*L*x**2))
'''
dqRe=(4*Dw)*dQ/(3.14*d*mu)
dRe=5*[dqRe]
#print 10*[dRe]

dqf=-(delH*g*(d**5)*(Dc-Dw)*(3.14**2))*dQ/(16*L*Dw*(Q**3))
dhf=(g*(d**5)*(Dc-Dw)*(3.14**2))*dH/(32*L*Dw*(Q**2))

#print dqf
#print dhf

df=scipy.sqrt(dqf**2+dhf**2)
dfl=df[:5]
dft=df[5:]

#print df

vel=Q*4/(3.14*(d**2))
#print vel
 
delP=delH*9.81*(Dc-Dw)
#print delP

Re=((4*Dw*Q)/(3.14*d*mu))
f=((delH*9.81*(d**5)*(3.14**2)*(Dc-Dw)/(32*L*Dw*(Q**2))))
Data={'Q(m^3/s)': Q,
        'delH(m)': delH,
        'vel(m/s)' : vel,
        'delP(Pa)' : delP,
        'Re': Re ,
        'sigRe':dqRe,
        'sigf':df,
        'f':f }
foot = pd.DataFrame(Data, columns=['Q(m^3/s)','delH(m)','delP(Pa)','Re','sigRe','f','sigf'])
print("-------------------------------------------------------------------------------------------")
print foot

Rel=Re[:5]
Ret=Re[5:]
fl=f[:5]
ft=f[5:]
laminar_data=foot[Re<4000]
print("-------------------------------------------------------------------------------------------")
print ('Laminar Data')
print laminar_data
#Rel=laminar_data(Re)
#fl=laminar_data(f)
#print Rel
print("-------------------------------------------------------------------------------------------")
tur_data=foot[Re>4000]
print ('Turbulent Data')
print tur_data
print("-------------------------------------------------------------------------------------------")
#Ret=tur_data(Re)
#ft=tur_data(f)
#print Rel
#print Ret
#print fl
#print ft
lnRe=scipy.log(Re)
lnRel=scipy.log(Rel)
lnRet=scipy.log(Ret)
lnf=scipy.log(f)
lnfl=scipy.log(fl)
lnft=scipy.log(ft)
#print lnRe
#print lnfl

def equation(x, A, B):
    return (A*x+B)

def residuals(p, y, x):
    A,B = p
    err = y-equation(x, A, B)
    return err

def peval(x, p):
    A,B = p
    return equation(x, A, B)

p0 = [-1, 3]

# Fit equation using least squares optimization
sol1 = leastsq(residuals, p0, args=(lnfl, lnRel))
P1=sol1[0]
#print P1[0],P1[1]
flfit=scipy.exp(P1[0]*lnRel+P1[1])
#print flfit
Relfit=scipy.exp((lnfl-P1[1])/P1[0])
#print Relfit

sol2 = leastsq(residuals, p0, args=(lnft, lnRet))
P2=sol2[0]
#print P2[0],P2[1]
ftfit=scipy.exp(P2[0]*lnRet+P2[1])
#print ftfit
Retfit=scipy.exp((lnft-P2[1])/P2[0])
#print Retfit

lnRec=-(P1[1]-P2[1])/(P1[0]-P2[0])
lnfc=P1[0]*lnRec+P1[1]
fc=scipy.exp(lnfc)
Rec=scipy.exp(lnRec)
#print Rec

plt.plot(lnRel,peval(lnRel,sol1[0]),lnRet,peval(lnRet,sol2[0]),lnRe,lnf,'o')
plt.legend(['Laminar','Turbulent','Expt'],loc='upper right')
plt.title('Least-squares fit to data')
plt.show()


pcov1=sol1[1]
pcov2=sol2[1]

rfl=((fl-flfit)/dfl)**2
#print rfl

rft=((ft-ftfit)/dft)**2
#print rft

rRel=((Rel-Relfit)/dRe)**2
#print rRel

rRet=((Ret-Retfit)/dRe)**2
#print rRet

def error_fit(Xdata,popt,pcov):
    Y=popt[0]*Xdata+popt[1]
    dY=[]
    for i in xrange(len(popt)):
        p=popt[i]
        dp=abs(p)/1e6+1e-20
        popt[i]+=dp
        Yi=popt[0]*Xdata+popt[1]
        dy=(Yi-Y)/dp
        dY.append(dy)
        popt[i]-=dp
        dY=scipy.array(dY)
        A=scipy.dot(dY.T,pcov)
        B=scipy.dot(A,dY)
        sigma2=B.diagonal()
        mean_sigma2=scipy.mean(sigma2)
        M=len(Xdata)
        N=len(popt)
        avg_stddev_data=scipy.sqrt(M*mean_sigma2/N)
        sigma=scipy.sqrt(sigma2)
        return sigma

sig1=error_fit(lnRel,P1,pcov1)
#print (0.1*sig1)
sig2=error_fit(lnRet,P2,pcov2)
#print (0.1*sig2)

M=len(Rel)
N=len(P1)

Relavg=scipy.mean(Rel)

squares=(Relfit-Relavg)
squaresT=(Rel-Relavg)
residuals=(Relfit-Rel)

SSM=sum(squares**2)
SSE=sum(residuals**2)
SST=sum(squaresT**2)

DFM=M-1
DFE=M-N
DFT=N-1

MSM=SSM/DFM
MSE=SSE/DFE
MST=SST/DFT

R2=SSM/SST
R2_adj=1-(1-R2)*(M-1)/(M-N-1)
print("-------------------------------------------------------------------------------------------")
print("Result of F Test")
print R2
print R2_adj

chisquared=sum(residuals**2)
Dof=M-N
chisquared_red=chisquared/Dof
p_chi2=1-scipy.stats.chi2.cdf(chisquared,Dof)
stderr_reg=scipy.sqrt(chisquared_red)
chisquare=(p_chi2,chisquared,chisquared_red,Dof,R2,R2_adj)
print("Chisquare Test Result")
print chisquare


w,p_shapiro=scipy.stats.shapiro(residuals)
mean_res=scipy.mean(residuals)
stddev_res=scipy.sqrt(scipy.var(residuals))
t_res=mean_res/stddev_res
p_res=1-scipy.stats.t.cdf(t_res,M-1)
print("Result Of Shapiro Residuals Test")
print p_res
 
F=MSM/MSE
p_F=1-scipy.stats.f.cdf(F,DFM,DFE)

print("Result Of F Test On Residuals")

dw=stools.durbin_watson(residuals)
print("Durbin Watson")
resanal=(p_shapiro,w,mean_res,p_res,F,p_F,dw)
print dw
print("-------------------------------------------------------------------------------------------")

Retavg=scipy.mean(Ret)

squares1=(Retfit-Retavg)
squaresT1=(Ret-Retavg)
residuals1=(Retfit-Ret)

SSM1=sum(squares1**2)
SSE1=sum(residuals1**2)
SST1=sum(squaresT1**2)

DFM=M-1
DFE=M-N
DFT=N-1

MSM1=SSM1/DFM
MSE1=SSE1/DFE
MST1=SST1/DFT

R21=SSM1/SST1
R2_adj1=1-(1-R21)*(M-1)/(M-N-1)

print("Result Of F Test")
print R21
print R2_adj1

chisquared1=sum(residuals1**2)
Dof=M-N
chisquared_red1=chisquared1/Dof
p_chi21=1-scipy.stats.chi2.cdf(chisquared1,Dof)
stderr_reg1=scipy.sqrt(chisquared_red1)
chisquare1=(p_chi21,chisquared1,chisquared_red1,Dof,R21,R2_adj1)
print("Chisquare Test Result")
print chisquare1

w1,p_shapiro1=scipy.stats.shapiro(residuals1)
mean_res1=scipy.mean(residuals1)
stddev_res1=scipy.sqrt(scipy.var(residuals1))
t_res1=mean_res1/stddev_res1
p_res1=1-scipy.stats.t.cdf(t_res1,M-1)
print("Result Of Shapiro Test")
print p_res1
 
F1=MSM1/MSE1
p_F1=1-scipy.stats.f.cdf(F1,DFM,DFE)

print("Result Of F Test on Residuals")

dw1=stools.durbin_watson(residuals1)
print("Durbin Watson")
resanal=(p_shapiro1,w1,mean_res1,p_res1,F1,p_F1,dw1)
print dw1
print("-------------------------------------------------------------------------------------------")


q=unumpy.uarray(Q,10*[10**-6])
#print q

h=unumpy.uarray(delH,10*[0.001])
#print h

Re=(4*Dw*q)/(3.14*d*mu)
f=(h*g*(Dc-Dw)*(d**5)*(3.14**2))/(32*L*Dw*(q**2))
#print Re
#print f
#Rem=unumpy.matrix(Re)
#fm=unumpy.matrix(f)

stddev=sum((unumpy.std_devs(f))**2)/2*M
#print stddev
ansf=ufloat(fc,stddev)
print ('Coeffs in Re vs f behaviour')
print ('Laminar Regime')
print scipy.exp(P1[1]),P1[0]
print ('Turbulent Regime')
print scipy.exp(P2[1]),P2[0]
print ('Critical value of friction factor is')
print ansf
Recric=(353.7*Dw*1.3*(10**-5)*3600/(0.1*(10**3)*d))          #http://zmixtech.com/mixguide/static/reynolds_num.html
print ('Theoretical Critical value of Reynolds number is' )
print Recric
ansRe=ufloat(Rec,dqRe)
print ('Critical value of Reynolds number is')
print ansRe
print("-------------------------------------------------------------------------------------------")