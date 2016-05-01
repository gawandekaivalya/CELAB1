# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 19:09:19 2016

@author: Kaivalya
"""


import numpy as np
import scipy
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import statsmodels.stats.stattools as stools
import win32com.client


'''
xl= win32com.client.gencache.EnsureDispatch("Excel.Application")
wb=xl.Workbooks('Data.xlsx')#.............work book Data,xlsx
sheet=wb.Sheets('Data')#..................sheet Data
   
def getdata(sheet, Range):
    data= sheet.Range(Range).Value
    data=scipy.array(data)
    data=data.reshape((1,len(data)))[0]
    return data

y_meas=getdata(sheet,"A3:A7")
'''
# Make up some data for fitting and add noise
# In practice, y_meas would be read in from a file
x = np.linspace(0,20,5)
A=-0.459
B=95.669
y_true=A*x+B
y_meas = y_true + 0.5*npr.randn(len(x))
#print y_meas
# Initial guess for parameters
p0 = [0, 100]

# Fit equation using least squares optimization
plsq = leastsq(y_meas-(A*x+B), p0)
P=plsq[0]
print P[0],P[1]
print A,B
y_fit=P[0]*x+P[1]
# Plot results
plt.plot(x,P[0]*x+P[1])
plt.title('Least-squares fit to data')
plt.show()

M=len(y_meas)
N=len(P)

y_mean=scipy.mean(y_meas)
print y_mean

squares=(y_fit-y_mean)
squaresT=(y_meas-y_mean)
residuals=(y_fit-y_meas)

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

print R2
print R2_adj


chisquared=sum(residuals**2)
Dof=M-N
chisquared_red=chisquared/Dof
p_chi2=1-scipy.stats.chi2.cdf(chisquared,Dof)
stderr_reg=scipy.sqrt(chisquared_red)
chisquare=(p_chi2,chisquared,chisquared_red,Dof,R2,R2_adj)
print chisquare

w,p_shapiro=scipy.stats.shapiro(residuals)
mean_res=scipy.mean(residuals)
stddev_res=scipy.sqrt(scipy.var(residuals))
t_res=mean_res/stddev_res
p_res=1-scipy.stats.t.cdf(t_res,M-1)
print p_res
if p_res<0.05:
    print ('Null Hypothesesis in rejected')


F=MSM/MSE
p_F=1-scipy.stats.f.cdf(F,DFM,DFE)
if p_F <0.05:
    print ('Null hypothesis is rejected')

dw=stools.durbin_watson(residuals)

resanal=(p_shapiro,w,mean_res,p_res,F,p_F,dw)