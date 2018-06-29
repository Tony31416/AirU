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
import scipy.stats, datetime, time
from matplotlib import cm

def ADC2R(ADC,Rpot=0):  #function to convert from ADC to Resistance
    return 2047*(820+Rpot)/ADC-(820+Rpot)

def hhmmss2min(hhmmss): #covert 'hh:mm:ss' string to minutes
    s=hhmmss.split(':')
    return int(s[0])*60+int(s[1])+int(s[2])/60  #convert to min

#path string to Data
path = "C:\\Users\\washi\\Google Drive\\AirU Folder\\scripts\\AirU (github)\\AirU\\Ozone\\O3Scoping\\O3_2018-06-13-11-00_C\\"
#path = "D:\\Google Drive\\AirU Folder\\scripts\\Examples\\Ozone Example\\"

f_all = os.listdir(path)  #read the directory
f=[]
for fname in f_all:  #loop throug all the files
    if (len(fname.split('.'))>1 and (fname.split('.')[1].lower()=='csv') and not ("O3AnalysisOutput" in fname) ): #only look for csv and ignore the fits file created by this code
        f.append(fname)
data={}     #create an empty dictionary
n = len(f)  #the number of files
itsAirU=[True]*n  #initialize the if AirU var
plt.close('all')   #close all the open python plots
fig1, ax1 = plt.subplots(2)  #open a new figure in which we will plot

OData = pd.DataFrame(columns = ['filename','fileID','SensorID','isAirU','isStandard']) #data to result from the analysis and become a csv file at the end
ThisData={}  #each new row of file data
#Let's load and condition the data in this folder...
fid=0 #file id
min_dt = float("inf")  #the smallest mode of the time step
max_t = -float("inf")  #the longest time
ExperimentData = {}  #place to keep all the miscillaneous info we want
for fname in f:  #loop throug all the files
    data[fid] = pd.read_csv( path + fname )   #load the cs
    ThisData['filename']=fname
    ThisData['fileID']=fid
    if ('CO' in data[fid].columns.values): # it's an AirU Sensor
        ThisData['isAirU']=True
        ThisData['isStandard']=False
        ThisData['SensorID']=data[fid]['ID'][0]
        data[fid]['O3']=ADC2R(data[fid]['NOX'])/1000 #converted to R in kohm
        data[fid]=data[fid][(data[fid]['O3']<1e2) & (data[fid]['O3']>.001)] #get rid of rediculous values
        data[fid]['t']=data[fid]['Timestamp'].apply(hhmmss2min)  #split the timestamp into hh mm and ss
        data[fid]['t']=data[fid]['t']-data[fid]['t'].min() #start at t=0
        ax1[0].plot(data[fid]['t'],data[fid]['O3'], label=fname)
        ax1[0].set_xlabel('Time (min)')
        ax1[0].set_ylabel('R (kOhm)')
    else: #not an AirU Sensor, it's our O3 sensor
        
        ThisData['isAirU']=False
        ThisData['isStandard']=True
        ThisData['SensorID']="106-L"
        O3standardID=fid  #the file id of the gold standard measuring device
        data[fid]=data[fid].rename(index=str,columns={"Cell Temp":"Temp", "Ozone":"O3"})  #make all data have same column heading for temperature
        data[fid]=data[fid][(data[fid]['O3']<1e3) & (data[fid]['O3']>.001)]  #get rid of rediculous values
        data[fid]['t']=data[fid]['Time'].apply(hhmmss2min)  #split the timestamp into hh mm and ss
        data[fid]['t']=data[fid]['t']-data[fid]['t'].min() #start at t=0
        ExperimentData['Start'] = data[fid]['Date'][0] + ' ' + data[fid]['Time'][0]  #date of experiment start
        try:
            ExperimentData['StartTimestamp'] = time.mktime(datetime.datetime.strptime(ExperimentData['Start'], "%d/%m/%y %H:%M:%S").timetuple())  #timestamp of experiment start
        except:
            ExperimentData['StartTimestamp'] = time.mktime(datetime.datetime.strptime(ExperimentData['Start'], "%d/%m/%Y %H:%M:%S").timetuple())
        ExperimentData['MeanCP'] = data[fid]['Cell Pressure'].mean() #cell pressure
        ExperimentData['StdCP'] = data[fid]['Cell Pressure'].std() #cell pressure
        ExperimentData['MeanQ'] = data[fid]['Flow Rate'].mean()           #Flowrate
        ExperimentData['StdQ'] = data[fid]['Flow Rate'].std()           #Flowrate
        ExperimentData['MeanPDV'] = data[fid]['PDV'].mean()           #Voltage for lamp
        ExperimentData['StdPDV'] = data[fid]['PDV'].std()           #Voltage for lamp
        ax1[1].plot(data[fid]['t'],data[fid]['O3'], 'k', label=fname)
        ax1[1].set_xlabel('Time (min)')
        ax1[1].set_ylabel('O3 (ppm)')
    
    data[fid].loc[data[fid]['Temp']<-10,'Temp']=data[fid]['Temp'].median() #fix temperatures
    
    mode_dt=sp.stats.mode(np.diff(data[fid]['t']))[0][0] #most frequent dt
    maxt=data[fid]['t'].max()
    if (min_dt>mode_dt): 
        min_dt=mode_dt  #the lowest most common dt
        min_dt_fid = fid  #the id of that dataset
    if (max_t<maxt):
        max_t=maxt
    print('{0:35s} - Size = {1:4d}x{2:1d},  Max t = {3:.0f} (min)'.format(fname,data[fid].shape[0],data[fid].shape[1],maxt))   #print the filename
    OData.loc[fid]=ThisData
    fid+=1 #incriment the file id
ax1[0].legend()
ax1[1].legend()
# Condition data to make it all in one dataframe
OData.sort_values('isStandard',ascending=False)  #sort the data to make sure our standard is first
OData['fileID'].tolist()
AirUIDs = OData[ OData['isStandard']!=True ]['fileID'].tolist()  #air U IDs
cols=['t','O3','T']  #column headings for new, conditioned data
for i in AirUIDs: #for airUs only
    cols.append('O3-'+str(i))   #ozone
    cols.append('T-'+str(i))    #temperature
    cols.append('H-'+str(i))    #humidity

cdata=pd.DataFrame(columns=cols)        #new set of conditioned data
cdata['t']=np.arange(0,maxt,min_dt)  #corrected data all on same timestep
lag=np.zeros(n)
fig2, ax2 = plt.subplots(len(data)+1)  #open a new figure in which we will plot

cdata['O3']=np.interp(cdata['t'],data[O3standardID]['t'],data[O3standardID]['O3']) #interpolate to make all data have same dt
cdata['T']=np.interp(cdata['t'],data[O3standardID]['t'],data[O3standardID]['Temp']) #temperature interpolated
x1=np.array(cdata['O3'])/np.max(np.array(cdata['O3'])) #normalize for visualization and xcorr
ax2[0].plot(cdata['t'],x1, 'k')   #plot normalized data
O3col, Tcol, Hcol, COcol = ['O3'],['T'],[],[]     #column headingsfor conditioned data
for k in AirUIDs: #loop thropugh the data
    O3col.append('O3-'+str(k))  #new column heading for ozone
    Tcol.append('T-'+str(k))    #column heading for temperature
    Hcol.append('H-'+str(k))    #column heading for humidity
    COcol.append('CO-'+str(k))    #column heading for humidity
    
    cdata[O3col[-1]]=np.interp(cdata['t'],data[k]['t'],data[k]['O3']) #interpolate to make all data have same dt
    cdata[Tcol[-1]]=np.interp(cdata['t'],data[k]['t'],data[k]['Temp']) #temperature interpolated
    iH = ((data[k]['Humidity']<100) & (data[k]['Humidity']>0)) #reasonable humidities
    cdata[Hcol[-1]]=np.interp(cdata['t'],data[k]['t'][iH],data[k]['Humidity'][iH]) #Humidity interpolated to make all data have same dt
    iCO = ((data[k]['CO']<1000) & (data[k]['CO']>0)) #reasonable humidities
    cdata[COcol[-1]]=np.interp(cdata['t'],data[k]['t'][iCO],data[k]['CO'][iCO]) #Humidity interpolated to make all data have same dt
    x2=np.array(cdata[O3col[-1]])/np.max(np.array(cdata[O3col[-1]]))  #normalized ozone
    ax2[0].plot(cdata['t'],x2, label=k)   #plot normalized data
    xcor = ax2[k].xcorr( x2, x1, maxlags = 2000, alpha=.5, color=[.3,.3,.3])#cross correlation
    lag[k]=xcor[0][np.argmax(xcor[1])] #how much time shift is needed for best correlation
    ax2[n].plot(cdata['t']-lag[k]*min_dt,x2, label=k) #plot with corrected time

ax2[n].plot(cdata['t']-lag[O3standardID]*min_dt,x1, 'k')   #plot normalized data

old_max_i = cdata['t'].shape[0]         #old length of data vectors
maxlag=np.max(lag)                      #max lag
newindex=[lag, old_max_i-maxlag+lag]    #new start and stop index for each dataset
new_max_i=int(old_max_i-maxlag)         #new length of data vectors
cdata['t']=cdata['t'][0:new_max_i]      #new time vector

for k,r in OData.iterrows():
    cdata[O3col[k]][0:new_max_i] = cdata[O3col[k]][int(newindex[0][k]):int(newindex[1][k])] #shift o3 data
    cdata[Tcol[k]][0:new_max_i] = cdata[Tcol[k]][int(newindex[0][k]):int(newindex[1][k])] #shift T data
    fid+=1
cdata=cdata.drop(range(new_max_i,old_max_i))  #get rid of extra data
cdata=cdata[['t']+O3col+Tcol+Hcol+COcol]            #rearrange columns just for looks

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
fig2.set_size_inches(12, 9)
fig2.savefig(path+'TimeSeries.png')
CL = .95 #confidence level
fig3, ax3 = plt.subplots(2,2)  #open a new figure in which we will plot
ax_list=[item for sublist in ax3 for item in sublist] 
fit_df = pd.DataFrame(columns = ["c1", "c1 ci", "c2", "c2 ci", "c3", "c3 ci", "R2"])
OtherData = pd.DataFrame(columns = ["", "MeanT", "StdT", "MeanH", "StdH", "MeanCO", "StdCO"])

for k in AirUIDs:
    ln, = ax3[0][0].plot(cdata['t'],cdata['O3-'+str(k)]/cdata['O3-'+str(k)].max(), label=k, alpha=.7) 
    ind = cdata['O3']>0
    x = cdata['O3-' + str(k)][ind]
    xfit = np.linspace(x.min(),x.max(),1000)
    y = cdata['O3'][ind]
    ax_list[k].plot(x, y, 'k.', alpha=.2) 
    ax_list[k].set_xlabel('Sensor R (KΩ)')
    ax_list[k].set_ylabel('O3 (ppm)')
    ax_list[k].grid('on')
    c_fit,R2,ci = fit_w_ci(x,y,linfitfun,CL)
    
    yfit=linfitfun(xfit,c_fit[0],c_fit[1],0)
    ax_list[k].plot(xfit, yfit, 'y--', alpha=.7) 

    c_fit = [100,0.5,-0.5]
    R2 = 0
    ci = [0,0,0]
    c_fit,R2,ci = fit_w_ci(x,y,nonlinfitfun,CL,p0=c_fit)
    j=0
    print('Line: {0:d}, y = c1*(x+c3)/(c2+x),  R2 = {1:10f}'.format(i,R2))
    for v in c_fit:
        print('C{0:d}: {1:10f} ± {2:10f} '.format(j+1, v, ci[j]) )
        j += 1
      #order from low to high
    yfit=nonlinfitfun(xfit,c_fit[0],c_fit[1],c_fit[2])
    ax_list[k].plot(xfit, yfit, 'r', alpha=.7) 

    fit_df.loc[k,["c1","c2","c3"]] = c_fit
    fit_df.loc[k,["c1 ci","c2 ci","c3 ci"]] = ci
    fit_df.loc[k,'R2'] = R2
OData=pd.concat([OData, fit_df], axis=1)    #add the fit data to the output

OData.to_csv(path+'O3AnalysisOutput.csv')   #save to file
fig3.set_size_inches(12, 9)
fig3.savefig(path+'Fits.png')

if (1==2):  #takes a lot of time and so can turn off by making if false
    cdata=cdata[['t']+O3col+Tcol+Hcol] #cdata=cdata[['t']+O3col+Tcol+Hcol+COcol]  get rid of what we don't want
    sax=pd.plotting.scatter_matrix(cdata, alpha=0.1, diagonal='kde', figsize =(12,12), s=5)  #make fancy scatter plot
    corr = cdata.corr().as_matrix()  #get correlation coeff
    for i, j in zip(*plt.np.triu_indices_from(sax, k=1)):  #format scatter plot to show corr
        sax[i, j].annotate("%.3f" %corr[i,j], (0.5, 0.2), xycoords='axes fraction', ha='center', va='center', backgroundcolor=(1,1,1))
        c=1-(corr[i,j]-corr.min())/(corr.max()-corr.min())
        sax[i, j].set_facecolor(cm.RdBu(c/2+.25))
        sax[j, i].set_facecolor(cm.RdBu(c/2+.25))
    plt.suptitle('Scatter Matrix')
    plt.savefig(path+'ScatterMatrix.png')