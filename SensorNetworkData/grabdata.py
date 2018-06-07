# -*- coding: utf-8 -*-
"""
Created on Mon May 21 13:38:30 2018

@author: tony3
"""

import requests
import pandas as pd
from urllib.parse import urlencode
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt  #plot library to show our data

TIMESTAMP = datetime.now().isoformat()

#################    PUT YOUR REQUEST HERE   ######################
sensIDs = ['S-A-101','S-A-114',130]               
start = '2018-05-19T00:00:00Z'
end = '2018-05-20T00:00:00Z'
###################################################################

DataThresholds = {    #bounds for min and max of each type of data
        'CO' : (0,1e4),  
        'H' : (0,100),  
        'T' : (-50,100),  
        'PM1' : (0,1e4),
        'PM2.5' : (0,1e4),
        'PM10' : (0,1e4)}

def getAQData(sensorIDs, start, end):
    data={}
    tags={}
    if end=='now':
        end=TIMESTAMP
    urlInfo={'id':'','sensorSource':'','start':start,'end':end,'show':'all'}
    #startDate = datetime.strptime(start, '%Y-%m-%dT%H:%M:%SZ')
    #endDate = datetime.strptime(end, '%Y-%m-%dT%H:%M:%SZ')
    for sid in sensorIDs: #loop through the sensor ids passes
        urlInfo['id'] = str(sid)
        try:
            int(sid)
            urlInfo['sensorSource']='Purple Air' #its a purple air sensor
        except ValueError:
            if 'S-A' in sid:
                urlInfo['sensorSource']='airu' #its a airu sensor
            else:
                urlInfo['sensorSource']='DAQ' #its a DAQ sensor (have to fix for mesowest)
        url =  'http://air.eng.utah.edu/dbapi/api/rawDataFrom?'+urlencode(urlInfo)
        print('######################### {0:s} #########################'.format(str(sid)))
        print(url)
        try:
            r=requests.get(url)  #get the data
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:  #possible errors
            print('\n{0:s}\tProblem acquiring historical data (HTTPError):'.format(TIMESTAMP))
            print(e)
            continue
        except requests.exceptions.Timeout as e:
            print('\n{0:s}\tProblem acquiring historical data (HTTPError):'.format(TIMESTAMP))
            print(e)
            continue
        except requests.exceptions.TooManyRedirects as e:
            print('\n{0:s}\tProblem acquiring historical data (HTTPError):'.format(TIMESTAMP))
            print(e)
            continue
        except requests.exceptions.RequestException as e:
            print('\n{0:s}\tProblem acquiring historical data (HTTPError):'.format(TIMESTAMP))
            print(e)
            continue
        data[sid] = r.json()['data']  #data collected
        tags[sid] = r.json()['tags']  #the tags
    return data,tags

data,tags = getAQData(sensIDs, start, end)
dataf={}
plt.close('all')
for sid in data:
    print('\n######################### {0:s} #########################'.format(str(sid)))
    dataf[sid]=pd.DataFrame.from_dict(data[sid])
    if (tags[sid][0]['Sensor Source']=='Purple Air'): #rename to same column headings
        dataf[sid]=dataf[sid].rename(index=str,columns={"Humidity (%)":"H", "Temp (*C)":"T", "pm1.0 (ug/m^3)":"PM1", "pm10.0 (ug/m^3)":"PM10", "pm2.5 (ug/m^3)":"PM2.5"}) 
    if (tags[sid][0]['Sensor Source']=='airu'):
        dataf[sid]=dataf[sid].rename(index=str,columns={"pm25":"PM2.5", "Temperature":"T", "Humidity":"H"}) 
    for i in DataThresholds:
        if (i in dataf[sid]):  #get rid of data above and below threshold
            dataf[sid][i][(dataf[sid][i]<DataThresholds[i][0]) | (dataf[sid][i]>DataThresholds[i][1])] = np.nan
    print(dataf[sid].describe())
    plt.plot(range(0,dataf[sid]['PM2.5'].count()) , dataf[sid]['PM2.5'])
    path = "D:\\Google Drive\\AirU Folder\\scripts\\Examples\\"
    dataf[sid].to_csv(path+'AQData-'+str(sid)+'.csv') 

 