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

def whatOutliers(x,y,ymin=0,ymax=1e4):  #code to remove outliers in data
#x is independent data, y is dependent, ymin is the minumin y could possibly be, ymax is the max
    nstd = 1.1
    w = 3 #window to look for big changes in dy
    avg = np.mean(y)
    std = np.std(y)
    ymax=min(ymax,avg+3*std)
    ymin=max(ymin,avg-3*std)
    i_keep = ( (y<ymax) & (y>ymin) ) #find where y is between min and max
    xtoss=list(x[np.invert(i_keep)])
    ytoss=list(y[np.invert(i_keep)])
    ynew = y[i_keep]     #keep only y-data between min and max
    xnew = x[i_keep]     #keep only x-data between min and max
    y=np.interp(x,xnew,ynew)  #replace the voids with interpolated data
    
    dy = np.diff(y) #derivative of data
    std = np.std(dy)  #stdev and mean of dy
    avg = np.mean(dy)
    dymin=avg-nstd*std  #window for max and min dy
    dymax=avg+nstd*std
    try:
        i_up = np.where(dy > dymax)[0]
        i_down = np.where(dy < dymin)[0]
        i_toss=np.array([]) 
        xnew, ynew = list(x), list(y)
        tossit=[]
        for up in i_up:
            down = i_down[np.where( (i_down < (up+w)) & (i_down > (up-w)))[0]]
            if len(down)>0:
                moves = np.append(down,up)
                i_toss = list( range(min(moves)+1,max(moves)+1) )
                for toss in i_toss:
                    tossit.append(toss)
        xnew=np.array(xnew) 
        ynew=np.array(ynew) 
        
        xnew=np.delete(xnew,tossit)
        ynew=np.delete(ynew,tossit)
        xtoss=xtoss+list(x[tossit])
        ytoss=ytoss+list(y[tossit])
        y=np.interp(x,xnew,ynew)  #replace the voids with interpolated data
    except:
        print('***   Outlier Removal Error   ***')
    return y,xtoss,ytoss

#path string to Data
path = "D:\\Google Drive\\AirU Folder\\scripts\\Examples\\Ozone Example\\run_5_23\\"
#path = "D:\\Google Drive\\AirU Folder\\scripts\\Examples\\Ozone Example\\"

f_all = os.listdir(path)  #read the directory
f=[]
for fname in f_all:  #loop throug all the files
    if (len(fname.split('.'))>1 and (fname.split('.')[1].lower()=='csv') and not ("O3AnalysisOutput" in fname) ): #only look for csv and ignore the fits file created by this code
        f.append(fname) #list of file names that contain data
data={}     #create an empty dictionary to hold all data
n = len(f)  #the number of data files
itsAirU=[True]*n  #initialize the if AirU var
plt.close('all')   #close all the open python plots

OData = pd.DataFrame(columns = ['filename','fileID','SensorID','HeatOn','N','RunTime']) #data to result from the analysis and become a csv file at the end
ThisData={}  #each new row of file data

##########   Let's load and condition the data in this folder...   ##########
fid=0 #file id
min_dt = float("inf")  #the smallest mode of the time step
max_t = -float("inf")  #the longest total timeof data collection
ExpData = {}  #place to keep all the miscillaneous info we want
AirU_fids = [] #airU dataset file ids (where they are in the loop through the files)
fig1, ax1 = plt.subplots(2,5)  #open a new figure in which we will plot

OutlierList = {'Temp':[-20,50,'T (°C)'], 'Humidity':[0,100,'H (%)'], 'CO':[0,1e3,'CO'], 'O3':[0,1e4,'(NOX-R kOhm)']}

for fname in f:  #loop throug all the files
    data[fid] = pd.read_csv( path + fname )   #load the cs
    if ('CO' in data[fid].columns.values): # it's an AirU Sensor
        AirU_fids.append(fid)
        ThisData['filename']=fname
        ThisData['fileID']=fid
        ThisData['SensorID']=data[fid]['ID'][0]
        ThisData['HeatOn']=data[0]['HEATER ON'].mean()  #is the heater on?
        ThisData['N']=data[0]['HEATER ON'].count()      #how many datum were collected?
        data[fid]['O3']=ADC2R(data[fid]['NOX'])/1000 #converted to R in kohm
        data[fid]['t']=data[fid]['Timestamp'].apply(hhmmss2min)  #split the timestamp into hh mm and ss
        data[fid]['t']=data[fid]['t']-data[fid]['t'].min() #start at t=0
        ThisData['RunTime']=data[0]['t'].max()      #how long was data collected?
        #plot raw data
        #data[fid]=data[fid][(data[fid]['O3']<1e2) & (data[fid]['O3']>.001)] #get rid of rediculous values
        ax1[0][0].plot(data[fid]['t'],data[fid]['Temp'],label=fname)
        ax1[0][0].set_xlabel('Time (min)'), ax1[0][0].set_ylabel('T (°C)')
        ax1[0][1].plot(data[fid]['t'],data[fid]['Humidity'],label=fname)
        ax1[0][1].set_xlabel('Time (min)'), ax1[0][1].set_ylabel('H (%)')
        ax1[0][2].plot(data[fid]['t'],data[fid]['CO'],label=fname)
        ax1[0][2].set_xlabel('Time (min)'), ax1[0][2].set_ylabel('CO')
        ax1[0][3].plot(data[fid]['t'],data[fid]['O3'],label=fname)
        ax1[0][3].set_xlabel('Time (min)'), ax1[0][3].set_ylabel('NOX-R (kOhm)')
        #remove outliers
        cntr=0
        for k, v in OutlierList.items():
            outlie = whatOutliers(data[fid]['t'],data[fid][k],v[0],v[1])
            data[fid][k] = outlie[0]
            ax1[1][cntr].plot(data[fid][k],label=fname)
            ax1[1][cntr].set_xlabel('Time (min)'), ax1[1][cntr].set_ylabel(v[2])
            ax1[0][cntr].plot(outlie[1],outlie[2],'rx')
            cntr+=1
        
        OData.loc[fid]=ThisData
    else: #not an AirU Sensor, it's our O3 sensor
        standID=fid  #the file id of the gold standard measuring device
        data[fid]=data[fid].rename(index=str,columns={"Cell Temp":"Temp", "Ozone":"O3"})  #make all data have same column heading for temperature
        data[fid]=data[fid][(data[fid]['O3']<1e3) & (data[fid]['O3']>.001)]  #get rid of rediculous values
        data[fid]['t']=data[fid]['Time'].apply(hhmmss2min)  #split the timestamp into hh mm and ss
        data[fid]['t']=data[fid]['t']-data[fid]['t'].min() #start at t=0
        ExpData['Start'] = data[fid]['Date'][0] + ' ' + data[fid]['Time'][0]  #date of experiment start
        try:
            ExpData['StartTimestamp'] = time.mktime(datetime.datetime.strptime(ExpData['Start'], "%d/%m/%y %H:%M:%S").timetuple())  #timestamp of experiment start
        except:
            ExpData['StartTimestamp'] = time.mktime(datetime.datetime.strptime(ExpData['Start'], "%d/%m/%Y %H:%M:%S").timetuple())
        ExpData['MeanCP'] = data[fid]['Cell Pressure'].mean() #cell pressure
        ExpData['StdCP'] = data[fid]['Cell Pressure'].std() #cell pressure
        ExpData['MeanQ'] = data[fid]['Flow Rate'].mean()           #Flowrate
        ExpData['StdQ'] = data[fid]['Flow Rate'].std()           #Flowrate
        ExpData['MeanPDV'] = data[fid]['PDV'].mean()           #Voltage for lamp
        ExpData['StdPDV'] = data[fid]['PDV'].std()           #Voltage for lamp
        ax1[0][0].plot(data[fid]['t'],data[fid]['Temp'],label=fname)
        ax1[0][0].set_xlabel('Time (min)'), ax1[0][1].set_ylabel('T (°C)')
        ax1[0][4].plot(data[fid]['t'],data[fid]['O3'], 'k', label=fname)
        ax1[0][4].set_xlabel('Time (min)'), ax1[0][4].set_ylabel('B2 O3 (ppb)')
    
    data[fid].loc[data[fid]['Temp']<-10,'Temp']=data[fid]['Temp'].median() #fix temperatures
    mode_dt=sp.stats.mode(np.diff(data[fid]['t']))[0][0] #most frequent dt
    maxt=data[fid]['t'].max()
    if (min_dt>mode_dt): 
        min_dt=mode_dt  #the lowest most common dt
        min_dt_fid = fid  #the id of that dataset
    if (max_t<maxt):
        max_t=maxt
    print('{0:35s} - Size = {1:4d}x{2:1d},  Max t = {3:.0f} (min)'.format(fname,data[fid].shape[0],data[fid].shape[1],maxt))   #print the filename
    fid+=1 #incriment the file id
ax1[0][4].legend()



###############   Condition data to make it all in one dataframe    ##############
cols=['t','O3','T']  #column headings for new, conditioned data
for k in AirU_fids: #for airUs only
    cols.append('O3-'+str(k))   #ozone
    cols.append('T-'+str(k))    #temperature
    cols.append('H-'+str(k))    #humidity

cdata=pd.DataFrame(columns=cols)        #new set of conditioned data
cdata['t']=np.arange(0,max_t,min_dt)    #corrected data all on same timestep
lag=np.zeros(n)                         #how for must we displace the data for best correlation with standard
fig2, ax2 = plt.subplots(len(data)+1)   #open a new figure in which we will plot

cdata['O3']=np.interp(cdata['t'],data[standID]['t'],data[standID]['O3']) #interpolate to make all data have same dt
cdata['T']=np.interp(cdata['t'],data[standID]['t'],data[standID]['Temp']) #temperature interpolated
x1=np.array(cdata['O3'])/np.max(np.array(cdata['O3'])) #normalize for visualization and xcorr
ax2[0].plot(cdata['t'],x1, 'k', label=f[standID])   #plot normalized data
O3col, Tcol, Hcol, COcol = ['O3'],['T'],[],[]     #column headings for conditioned data
axi=1 
for k in AirU_fids: #loop thropugh the data
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
    ax2[0].plot(cdata['t'],x2, label=f[k])   #plot normalized data
    xcor = ax2[axi].xcorr( x2, x1, maxlags = 2000, alpha=.5, color=[.3,.3,.3])#cross correlation
    lag[axi]=xcor[0][np.argmax(xcor[1])] #how much time shift is needed for best correlation
    ax2[n].plot(cdata['t']-lag[axi]*min_dt,x2, label=f[k]) #plot with corrected time
    axi+=1

ax2[n].plot(cdata['t']-lag[0]*min_dt,x1, 'k', label=f[standID])   #plot normalized data
ax2[0].legend()
ax2[n].legend()

old_max_i = cdata['t'].shape[0]         #old length of data vectors
maxlag=np.max(lag)                      #max lag
newindex=[lag, old_max_i-maxlag+lag]    #new start and stop index for each dataset
new_max_i=int(old_max_i-maxlag)         #new length of data vectors
cdata['t']=cdata['t'][0:new_max_i]      #new time vector

for k in range(0,n):
    cdata[O3col[k]][0:new_max_i] = cdata[O3col[k]][int(newindex[0][k]):int(newindex[1][k])] #shift o3 data
    cdata[Tcol[k]][0:new_max_i] = cdata[Tcol[k]][int(newindex[0][k]):int(newindex[1][k])] #shift T data
    #if (k>0):
    #cdata[Hcol[k-1]][0:new_max_i] = cdata[Hcol[k-1]][int(newindex[0][k]):int(newindex[1][k])] #shift T data
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
fit_df = pd.DataFrame(columns = ["c1", "ci1", "c2", "ci2", "c3", "ci3", "R2"])
OtherData = pd.DataFrame(columns = ["MeanT", "StdT", "MeanH", "StdH", "MeanCO", "StdCO"]+list(ExpData.keys()))
axi=1
ax3[0][0].plot(cdata['t'],cdata['O3']/cdata['O3'].max(), label=f[standID], alpha=.7, c='k') 
for k in AirU_fids:
    ln, = ax3[0][0].plot(cdata['t'],cdata['O3-'+str(k)]/cdata['O3-'+str(k)].max(), label=f[k], alpha=.7) 
    ind = cdata['O3']>0
    x = cdata['O3-' + str(k)][ind]
    xfit = np.linspace(x.min(),x.max(),1000)
    y = cdata['O3'][ind]
    ax_list[axi].plot(x, y, 'k.', alpha=.2, label=f[k]) 
    ax_list[axi].set_xlabel('Sensor R (KΩ)')
    ax_list[axi].set_ylabel('O3 (ppb)')
    ax_list[axi].grid('on')
    
    c_fit,R2,ci = fit_w_ci(x,y,linfitfun,CL)
    
    yfit=linfitfun(xfit,c_fit[0],c_fit[1],0)
    ax_list[axi].plot(xfit, yfit, 'y--', alpha=.7, label = "Lin Fit") 

    c_fit = [100,0.5,-0.5]
    R2 = 0
    ci = [0,0,0]
    c_fit,R2,ci = fit_w_ci(x,y,nonlinfitfun,CL,p0=c_fit)
    j=0
    print('{0:s}: y = c1*(x+c3)/(c2+x),  R2 = {1:10f}'.format(f[k],R2))
    for v in c_fit:
        print('C{0:d}: {1:10f} ± {2:10f} '.format(j+1, v, ci[j]) )
        j += 1
      #order from low to high
    yfit=nonlinfitfun(xfit,c_fit[0],c_fit[1],c_fit[2])
    ax_list[axi].plot(xfit, yfit, 'r', alpha=.7, label = "Langmuir Fit") 
    ax_list[axi].legend()
    fit_df.loc[k,["c1","c2","c3"]] = c_fit
    fit_df.loc[k,["ci1","ci2","ci3"]] = ci
    fit_df.loc[k,'R2'] = R2
    OtherData.loc[k,'MeanT']=cdata['T-' + str(k)].mean()
    OtherData.loc[k,'StdT']=cdata['T-' + str(k)].std()
    OtherData.loc[k,'MeanH']=cdata['H-' + str(k)].mean()
    OtherData.loc[k,'StdH']=cdata['H-' + str(k)].std()
    OtherData.loc[k,'MeanCO']=cdata['CO-' + str(k)].mean()
    OtherData.loc[k,'StdCO']=cdata['CO-' + str(k)].std()
    OtherData.loc[k,list(ExpData.keys()) ] = ExpData
    axi += 1
ax3[0][0].legend()
OData=pd.concat([OData, fit_df], axis=1)    #add the fit data to the output
OData=pd.concat([OData, OtherData], axis=1)#add the other data to the output
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