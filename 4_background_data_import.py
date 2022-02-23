#!/usr/bin/env python
# coding: utf-8
#

import pandas as pd
import matplotlib as mb
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from datetime import datetime
from bisect import bisect

# load DBSCAN wind farms data
fp=pd.read_csv('../data/io_cluster.csv')
loca=pd.DataFrame(fp,columns=['xlong','ylat'])
col_cap=['p_tnum','p_cap','t_cap']
cap=pd.DataFrame(fp,columns=col_cap)
xl=[]
yl=[]
z=[]
gr=fp.groupby('cluster')

#capacity
def cal_capacity():
    xl=[]
    yl=[]
    z=[]
    for i in range(0,419):
        p_tnum=gr.get_group(i)['p_tnum'].mode().mean()
        p_cap=gr.get_group(i)['p_cap'].mode().mean()
        t_cap=gr.get_group(i)['t_cap'].mode().mean()
        xl.append(p_tnum)
        yl.append(p_cap)
        z.append(t_cap)
    capacity=pd.concat([pd.DataFrame(xl),pd.DataFrame(yl),pd.DataFrame(z)],axis=1)  
    capacity.columns=['p_tnum','p_cap','t_cap']
    return capacity
cal_capacity()

#real  capacity
def cal_real_capacity():
    cap=pd.DataFrame(fp,columns=['cluster','p_tnum','t_cap'])
    #set nan to normal 1500
    cap['t_cap'][cap['t_cap']<0]=1500
    group=cap.groupby('cluster')
    capacity_real=[]
    for i in range(0,419):
        total_cap=group.get_group(i)['t_cap'].sum()/1000  #KW-MW
        capacity_real.append(total_cap)
    cap_real=pd.DataFrame(capacity_real)
    cap_real.columns=['cap_real']
    return cap_real
cal_real_capacity()

#location
def cal_location():
    xl=[]
    yl=[]
    z=[]
    for i in range(0,419):
        x=gr.get_group(i)['xlong'].mode().mean()
        y=gr.get_group(i)['ylat'].mode().mean()
        xl.append(x)
        yl.append(y)
    location=pd.concat([pd.DataFrame(xl),pd.DataFrame(yl)],axis=1)  
    location.columns=['xlong','ylat']
    return location
cal_location()

def cal_construction_year():
    box=[]
    for i in range(0,419):
        bui = gr['p_year'].get_group(i).mode()
        box.append(bui)
    b_year = pd.DataFrame(box)
    b_year[1]=b_year[0]
    b_year[1][b_year[1] < 0] = 2004
    b_year[1][b_year[1] < 2004] = 2004
    b_year[1][b_year[1] > 2017] = 2017
    b_year.columns=['build_year_most_original','built_year_most_for_calculation']
    return b_year

location=cal_location()
build_year=cal_construction_year()
capacity = cal_capacity()
capacity_real=cal_real_capacity()


tot_all=pd.concat([location,build_year,capacity,cap_real],axis=1)
tot_all   # complete 419 wind farms

#ndvi
wf_ndvi_aqua=pd.read_csv('../data/US-wf/wf_ndvi_aqua.csv').iloc[:,1:4]
x2=wf_ndvi_aqua['date']
#将gee输出格式的时间索引转化为Python可识别的时间索引
y2=[]
for i in wf_ndvi_aqua['date']:
    datetime_object = datetime.strptime(i, '%Y_%m_%d')
    dti=pd.to_datetime(datetime_object)
    y2.append(dti)
y2=pd.DataFrame(y2)
y2.columns=['Date']
ndvi=pd.concat([y2,wf_ndvi_aqua],axis=1)
ndvi=ndvi.set_index('Date')
ndvi=ndvi.drop(['date'],axis=1)
nd=ndvi.groupby('cluster')

nd_mean=[]
for i in range(0,419):
    year=str(int(tot_all['built_year_most_for_calculation'][i]))
    ndvi_mean=nd.get_group(i)[year]['mean'].mean()*0.0001
    nd_mean.append(ndvi_mean)
nd_mean=pd.DataFrame(nd_mean,columns=['ndvi'])

nd_max=[]
for i in range(0,419):
    year=str(int(tot_all['built_year_most_for_calculation'][i]))
    ndvi_=pd.DataFrame(nd.get_group(i)[year]['mean'])
    ndvi_max=ndvi_['mean'].quantile(0.95)*0.0001
    nd_max.append(ndvi_max)
nd_max=pd.DataFrame(nd_max,columns=['ndvi_95'])
#nd_max.to_csv('/mnt/e/US/ndvi_95.csv')

#
lc=pd.read_csv('../data/background/background_land_cover.csv').iloc[:,2:4]
lc.sort_values('cluster',inplace=True)
land=lc.set_index('cluster')[1:420]
land.columns=['land_cover']
land['land_cover'].value_counts()

#real built year
bui=bui_year
bui[bui < 0] = 2004
bui[bui < 2004] = 2004
bui[bui > 2017] = 2018
bui_y=pd.DataFrame(bui['0'])
bui_y.columns=['build_year_real']


# precipitation & elevation
pre=pd.read_csv('../data/background/precipitation.csv').iloc[:,2:4]
ele=pd.read_csv('../data/background/elevation.csv').iloc[:,2:4]
pre.sort_values('cluster',inplace=True)
precip=pre.set_index('cluster')[1:420]
precip.columns=['precipitation']
#to monthly
precip=30*24*precip
ele.sort_values('cluster',inplace=True)
elevation=ele.set_index('cluster')[1:420]
elevation.columns=['elevation']

ele_nwf=pd.read_csv('../data/background/elevation_nwf.csv').iloc[:,5:7]
ele_nwf=ele_nwf.set_index('cluster').sort_index()
ele_nwf=ele_nwf[1:420]
ele_minus=pd.DataFrame(elevation['elevation']-ele_nwf['mean'],columns=['ele_minus'])

tot_all_land=pd.concat([tot_all,land,precip,elevation,ele_minus,bui_y,nd_mean],axis=1)
tot_all_land
#tot_all_land.to_csv('background_data.csv')
