# -*- coding: utf-8 -*-
"""
Created on Tue May 15 09:57:05 2018

@author: tony3
"""

#import the packages I'll need

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats

def ADC2R(ADC,Rpot=0):  #function to convert from ADC to Resistance
    return 2047*(820+Rpot)/ADC-(820+Rpot)

def hhmmss2min(hhmmss): #covert 'hh:mm:ss' string to minutes
    s=hhmmss.split(':')
    return int(s[0])*60+int(s[1])+int(s[2])/60  #convert to min

#path string to Data
path = "D:\\Google Drive\\AirU Folder\\scripts\\Examples\\Ozone Example\\"

f_all = os.listdir(path)  #read the directory
f=[]
for fname in f_all:  #loop throug all the files
    if ( (fname.split('.')[1].lower()=='csv') and not ("fits" in fname) ):
        f.append(fname)
data={}     #create an empty dictionary
n = len(f)  #the number of files
itsAirU=[True]*n  #initialize the if AirU var
plt.close('all')   #close all the open python plots
fig1, ax1 = plt.subplots(2)  #open a new figure in which we will plot

#Let's load and condition the data in this folder...
fid=0 #file id
min_dt = float("inf")  #the smallest mode of the time step
max_t = -float("inf")  #the longest time
for fname in f:  #loop throug all the files
    if (fname.split('.')[1].lower()=='csv'):  #if it's a csv file
        data[fname] = pd.read_csv(path+fname)   #load the cs
        if ('CO' in data[fname].columns.values): # it's an AirU Sensor
            itsAirU[fid]=True
            data[fname]['O3']=ADC2R(data[fname]['NOX'])/1000 #converted to R in kohm
            data[fname]=data[fname][(data[fname]['O3']<1e2) & (data[fname]['O3']>.001)] #get rid of rediculous values
            data[fname]['t']=data[fname]['Timestamp'].apply(hhmmss2min)  #split the timestamp into hh mm and ss
            data[fname]['t']=data[fname]['t']-data[fname]['t'].min() #start at t=0
            ax1[0].plot(data[fname]['t'],data[fname]['O3'], label=fname)
            ax1[0].set_xlabel('Time (min)')
            ax1[0].set_ylabel('R (kOhm)')
        else: #not an AirU Sensor, it's our O3 sensor
            itsAirU[fid]=False
            data[fname]=data[fname].rename(index=str,columns={"Cell Temp":"Temp", "Ozone":"O3"})  #make all data have same column heading for temperature
            data[fname]=data[fname][(data[fname]['O3']<1e3) & (data[fname]['O3']>.001)]
            data[fname]['t']=data[fname]['Time'].apply(hhmmss2min)  #split the timestamp into hh mm and ss
            data[fname]['t']=data[fname]['t']-data[fname]['t'].min() #start at t=0
            ax1[1].plot(data[fname]['t'],data[fname]['O3'], label=fname)
            ax1[1].set_xlabel('Time (min)')
            ax1[1].set_ylabel('O3 (ppm)')
        
        data[fname].loc[data[fname]['Temp']<-10,'Temp']=data[fname]['Temp'].median() #fix temperatures
        
        mode_dt=sp.stats.mode(np.diff(data[fname]['t']))[0][0] #most frequent dt
        maxt=data[fname]['t'].max()
        if (min_dt>mode_dt): 
            min_dt=mode_dt  #the lowest most common dt
            min_dt_fid = fid  #the id of that dataset
        if (max_t<maxt):
            max_t=maxt
        fid+=1 #incriment the file id
        print('{0:35s} - Size = {1:4d}x{2:1d},  Max t = {3:.0f} (min)'.format(fname,data[fname].shape[0],data[fname].shape[1],maxt))   #print the filename

# Condition data to make it all in one dataframe
cols=['t']
for i in range(fid):
    cols.append('O3-'+str(i))
    cols.append('T-'+str(i))

cdata=pd.DataFrame(columns=cols)
cdata['t']=np.arange(0,maxt,min_dt)
fid=0
lag=np.zeros(n)
fig2, ax2 = plt.subplots(5)  #open a new figure in which we will plot
for k in data: #loop thropugh the data
    O3key='O3-'+str(fid)  #new column heading for ozone
    Tkey='T-'+str(fid)    #column heading for temperature
    cdata[O3key]=np.interp(cdata['t'],data[k]['t'],data[k]['O3']) #interpolate to make all data have same dt
    cdata[Tkey]=np.interp(cdata['t'],data[k]['t'],data[k]['Temp']) #temperature
    
    x1=np.array(cdata['O3-0'])/np.max(np.array(cdata['O3-0'])) #normalize for visualization and xcorr
    x2=np.array(cdata[O3key])/np.max(np.array(cdata[O3key]))
    ax2[0].plot(cdata['t'],x2, label=k)   #plot normalized data
    if fid!=0:  #get cross correlation with first dataset if not the first dataset
        xcor = ax2[fid].xcorr( x2, x1, maxlags = 2000, alpha=.5, color=[.3,.3,.3])#cross correlation
        lag[fid]=xcor[0][np.argmax(xcor[1])] #how much time shift is needed for best correlation
    else: lag[fid]=0 #can't lag with self...
    ax2[n].plot(cdata['t']-lag[fid]*min_dt,x2, label=k) #plot with corrected time
    fid+=1
#lag[2]=lag[1]

old_max_i = cdata['t'].shape[0]   #old length of data vectors
maxlag=np.max(lag)  #max lag
newindex=[lag, old_max_i-maxlag+lag]
new_max_i=int(old_max_i-maxlag)   #new length of data vectors
cdata['t']=cdata['t'][0:new_max_i]   #new time vector

O3col, Tcol = [],[]
for k in range(0,n):
    O3col.append('O3-'+str(k))  #new column heading for ozone
    Tcol.append('T-'+str(k))    #column heading for temperature
    cdata[O3col[k]][0:new_max_i]=cdata[O3col[k]][int(newindex[0][k]):int(newindex[1][k])] #shift o3 data
    cdata[Tcol[k]][0:new_max_i]=cdata[Tcol[k]][int(newindex[0][k]):int(newindex[1][k])] #shift T data
    fid+=1
cdata=cdata.drop(range(new_max_i,old_max_i))  #get rid of extra data
cdata=cdata[['t']+O3col+Tcol]    #rearrange columns just for looks

if (1==2):
    sax=pd.plotting.scatter_matrix(cdata, alpha=0.1, diagonal='kde')  #make fancy scatter plot
    corr = cdata.corr().as_matrix()  #get correlation coeff
    for i, j in zip(*plt.np.triu_indices_from(sax, k=1)):  #format scatter plot to show corr
        sax[i, j].annotate("%.3f" %corr[i,j], (0.5, 0.2), xycoords='axes fraction', ha='center', va='center')
        c=(corr[i,j]-corr.min())/(corr.max()-corr.min())
        sax[i, j].set_facecolor([1, 1, 1-c])
        sax[j, i].set_facecolor([1, 1, 1-c])
    plt.suptitle('Scatter Matrix')


################## fit a line #######################
def fit_w_ci(x,y,fitfun,cl=0.95,p0 = [0,0,0]):  #take in x y data, fint fit, and give conf intervals
    c_fit, c_cov = sp.optimize.curve_fit(fitfun, x, y,p0=p0) #get fit constants and covariance matrix
    yfit=fitfun(x,c_fit[0],c_fit[1],c_fit[2])  #get the trendline values for y
    res = y - yfit  #find the  residuals
    ss_res = np.sum(res**2)  #sum of squares or res
    ss_tot = np.sum( ( y - np.mean(y) )**2 ) #total sum of squares
    R2 = 1 - (ss_res / ss_tot) #Rsquarred value, coefficient of determination

    d_cov = np.diag(c_cov)  #diagonal of the covariance matrix
    alpha = 1 - cl  # 1 - Conf level
    dof = max(0, len(y)-len(c_fit)) #degrees of freedom
    t_val = sp.stats.distributions.t.ppf(1-alpha/2, dof)  #student t test stat
    ci = (d_cov**.5)*t_val   #confidence interval
    return c_fit,R2,ci

def linfitfun(x,m,b,c):   #function to fit to, x must be first
    return m*x+b   #just a straight line
def nonlinfitfun(x,c1,c2,c3):   #function to fit to, x must be first
    return c1*(x+c3)/(c2+x)  #langmuir isotherm

CL = .95 #confidence level
fig3, ax3 = plt.subplots(2,2)  #open a new figure in which we will plot
ax_list=[item for sublist in ax3 for item in sublist] 
fit_df = pd.DataFrame(columns = ["datafile", "c1", "c1 ci", "c2", "c2 ci", "c3", "c3 ci", "R2"])
for i in range(0,n):
    ln, = ax3[0][0].plot(cdata['t'],cdata['O3-'+str(i)]/cdata['O3-'+str(i)].max(), label=k, alpha=.7) 
    if (i!=0):  #it's not the gold standard device
        ind=cdata['O3-0']>0
        x=cdata['O3-'+str(i)][ind]
        y=cdata['O3-0'][ind]
        ax_list[i].plot(x, y, 'k.', alpha=.2) 
        ax_list[i].set_xlabel('Sensor R (KΩ)')
        ax_list[i].set_ylabel('O3 (ppm)')
        ax_list[i].grid('on')
        c_fit,R2,ci = fit_w_ci(x,y,linfitfun,CL)
        yfit=linfitfun(x,c_fit[0],c_fit[1],0)
        ax_list[i].plot(x, yfit, 'y--', alpha=.7) 

        c_fit=[100,0.5,-0.5]
        R2=0
        ci=[0,0,0]
        c_fit,R2,ci = fit_w_ci(x,y,nonlinfitfun,CL,p0=c_fit)
        k=0
        print('Line: {0:d}, y = c1*(x+c3)/(c2+x),  R2 = {1:10f}'.format(i,R2))
        for v in c_fit:
            print('C{0:d}: {1:10f} ± {2:10f} '.format(k+1, v, ci[k]) )
            k+=1
        xfit=sorted(x)  #order from low to high
        yfit=nonlinfitfun(xfit,c_fit[0],c_fit[1],c_fit[2])
        ax_list[i].plot(xfit, yfit, 'r', alpha=.7) 
        fit_df.loc[i-1,"datafile"] = f[i]
        fit_df.loc[i-1,["c1","c2","c3"]] = c_fit
        fit_df.loc[i-1,["c1 ci","c2 ci","c3 ci"]] = ci
        fit_df.loc[i-1,'R2'] = R2
    else:
        ln.set_c([0,0,0])
fit_df.to_csv(path+'fits.csv')