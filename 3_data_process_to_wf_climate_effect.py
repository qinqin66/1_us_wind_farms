#!/usr/bin/env python
# coding: utf-8

# import all the lib needed
import pandas as pd
import matplotlib as mb
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels
import numpy as np
from numpy.linalg import inv #逆矩阵
from numpy import dot #矩阵乘
from numpy import mat #矩阵
from sklearn import linear_model
import numpy as np
reg = linear_model.LinearRegression()
import seaborn as sns
import scipy

#read raw data
wf_lst_aqua_day=pd.read_csv('../data/3_data_zonal_stat_gee/aqua/wf_lst_aqua_day.csv').iloc[:,1:4]
wf_lst_aqua_night=pd.read_csv('../data/3_data_zonal_stat_gee/aqua/wf_lst_aqua_night.csv').iloc[:,1:4]

wf_lst_aqua_day_8_10=pd.read_csv('../data/3_data_zonal_stat_gee/aqua/nwf8_10_lst_aqua_day.csv').iloc[:,1:4]
wf_lst_aqua_night_8_10=pd.read_csv('../data/3_data_zonal_stat_gee/aqua/nwf8_10_lst_aqua_night.csv').iloc[:,1:4]

wf_lst_aqua_day_6_8=pd.read_csv('../data/3_data_zonal_stat_gee/aqua/nwf6_8_lst_aqua_day.csv').iloc[:,1:4]
wf_lst_aqua_night_6_8=pd.read_csv('../data/3_data_zonal_stat_gee/aqua/nwf6_8_lst_aqua_night.csv').iloc[:,1:4]

wf_lst_aqua_day_4_6=pd.read_csv('../data/3_data_zonal_stat_gee/aqua/nwf4_6_lst_aqua_day.csv').iloc[:,1:4]
wf_lst_aqua_night_4_6=pd.read_csv('../data/3_data_zonal_stat_gee/aqua/nwf4_6_lst_aqua_night.csv').iloc[:,1:4]

wf_lst_aqua_day_2_4=pd.read_csv('../data/3_data_zonal_stat_gee/aqua/nwf2_4_lst_aqua_day.csv').iloc[:,1:4]
wf_lst_aqua_night_2_4=pd.read_csv('../data/3_data_zonal_stat_gee/aqua/nwf2_4_lst_aqua_night.csv').iloc[:,1:4]

x=wf_lst_aqua_night_8_10['date']
#将gee输出格式的时间索引转化为Python可识别的时间索引
y=[]
for i in wf_lst_aqua_night_8_10['date']:
    datetime_object = datetime.strptime(i, '%Y_%m_%d')
    dti=pd.to_datetime(datetime_object)
    y.append(dti)
y=pd.DataFrame(y)
y.columns=['Date']
a=pd.concat([y,wf_lst_aqua_night],axis=1)
a=a.set_index('Date')
a=a.drop(['date'],axis=1)
b=pd.concat([y,wf_lst_aqua_night_8_10],axis=1)
b=b.set_index('Date')
b=b.drop(['date'],axis=1)

def daily_to_monthly(x):
    b=pd.concat([y,x],axis=1)
    b=b.set_index('Date')
    b=b.drop(['date'],axis=1)
    nwf=b.groupby('cluster')
    nwf_mean=[]
    nwf_night=pd.DataFrame(nwf_mean)
    for i in range(0,419):
        nwf_0=nwf.get_group(i).resample('MS').mean()
        nwf_1=pd.DataFrame(nwf_0['mean'])
        nwf_1.columns=[i]
        nwf_night=pd.concat([nwf_night,nwf_1],axis=1)
    nwf_night=nwf_night*0.02-273.15
    return nwf_night

wf_night=daily_to_monthly(wf_lst_aqua_night)
nwf_2_4_night=daily_to_monthly(wf_lst_aqua_night_2_4)
nwf_4_6_night=daily_to_monthly(wf_lst_aqua_night_4_6)
nwf_6_8_night=daily_to_monthly(wf_lst_aqua_night_6_8)
nwf_8_10_night=daily_to_monthly(wf_lst_aqua_night_8_10)

wf_day=daily_to_monthly(wf_lst_aqua_day)
nwf_2_4_day=daily_to_monthly(wf_lst_aqua_day_2_4)
nwf_4_6_day=daily_to_monthly(wf_lst_aqua_day_4_6)
nwf_6_8_day=daily_to_monthly(wf_lst_aqua_day_6_8)
nwf_8_10_day=daily_to_monthly(wf_lst_aqua_day_8_10)

#series of WF minus NWF
wf_nwf_8_10_night=wf_night-nwf_8_10_night
wf_nwf_6_8_night=nwf_6_8_night - nwf_8_10_night
wf_nwf_4_6_night=nwf_4_6_night - nwf_8_10_night
wf_nwf_2_4_night=nwf_2_4_night - nwf_8_10_night

wf_nwf_8_10_day=wf_day - nwf_8_10_day
wf_nwf_6_8_day=nwf_6_8_day - nwf_8_10_day
wf_nwf_4_6_day=nwf_4_6_day - nwf_8_10_day
wf_nwf_2_4_day=nwf_2_4_day - nwf_8_10_day


# wf_nwf_2_4_day.to_csv('/mnt/e/US/data/result/wf_nwf_2_4_day.csv')
# wf_nwf_4_6_day.to_csv('/mnt/e/US/data/result/wf_nwf_4_6_day.csv')
# wf_nwf_6_8_day.to_csv('/mnt/e/US/data/result/wf_nwf_6_8_day.csv')
# wf_nwf_8_10_day.to_csv('/mnt/e/US/data/result/wf_nwf_8_10_day.csv')

# wf_nwf_2_4_night.to_csv('/mnt/e/US/data/result/wf_nwf_2_4_night.csv')
# wf_nwf_4_6_night.to_csv('/mnt/e/US/data/result/wf_nwf_4_6_night.csv')
# wf_nwf_6_8_night.to_csv('/mnt/e/US/data/result/wf_nwf_6_8_night.csv')
# wf_nwf_8_10_night.to_csv('/mnt/e/US/data/result/wf_nwf_8_10_night.csv')

build=pd.read_csv('/mnt/e/US/data/background/background_info_whole.csv')
build_year=pd.DataFrame(build['wf_most_built_year_for_calculation'])
build_year.columns=['build_year']


#monthly WF - NWF
def month_minus(wf_nwf_8_10_night,mon):
    box=[]
    for i in range(0,419):
        data=wf_nwf_8_10_night[i]
        year=int(build_year.loc[i])
        whi=str(int(year))
        str_mon=str("-"+str(mon))
        if year==2004:
            mid=str(int(year))+str_mon
            after1=str(int(year+1))+str_mon
            after2=str(int(year+2))+str_mon
            before1=str(int(year-1))+str_mon
            mon_minus=(float(data[after1])+float(data[after2])+float(data[mid]))/3-(float(data[mid])+float(data[before1]))/2
            box.append(mon_minus)
        elif year>2004 and year<2017:
            mid=str(int(year))+str_mon
            after1=str(int(year+1))+str_mon
            after2=str(int(year+2))+str_mon
            before1=str(int(year-1))+str_mon
            before2=str(int(year-2))+str_mon
            mon_minus=(float(data[after1])+float(data[after2])+float(data[mid]))/3-(float(data[mid])+float(data[before1])+float(data[before2]))/3
            box.append(mon_minus)
        elif year==2017:
            mid=str(int(year))+str_mon
            after1=str(int(year+1))+str_mon
            before2=str(int(year-2))+str_mon
            before1=str(int(year-1))+str_mon
            mon_minus=(float(data[after1])+float(data[mid]))/2-(float(data[mid])+float(data[before1])+float(data[before2]))/2
            box.append(mon_minus)
    return box

#monthly trend 
def month_trend(wf_nwf_8_10_night,month):
    box=[]
    for i in range(0,419):
        data=wf_nwf_8_10_night[i]
        year=int(build_year.loc[i])
        whi=str(int(year))
        str_mon=str("-"+str(month))
        if year==2004:
            x=np.array([1,2,3,4])
            mid=str(int(year))+str_mon
            after1=str(int(year+1))+str_mon
            after2=str(int(year+2))+str_mon
            before1=str(int(year-1))+str_mon
            series=pd.DataFrame(np.array([float(data[before1]),float(data[mid]),float(data[after1]),float(data[after2])]))
            y=series.fillna(series.mean())
            tem_trend,r=np.polyfit(x,np.nan_to_num(y),1)
            box.append(tem_trend)
        elif year>2004 and year<2017:
            x=np.array([1,2,3,4,5])
            mid=str(int(year))+str_mon
            after1=str(int(year+1))+str_mon
            after2=str(int(year+2))+str_mon
            before1=str(int(year-1))+str_mon
            before2=str(int(year-2))+str_mon
            series=pd.DataFrame(np.array([float(data[before2]),float(data[before1]),float(data[mid]),float(data[after1]),float(data[after2])]))
            y=series.fillna(series.mean())
            tem_trend,r=np.polyfit(x,np.nan_to_num(y),1)
            box.append(tem_trend)
        elif year==2017:
            x=np.array([1,2,3,4])
            mid=str(int(year))+str_mon
            after1=str(int(year+1))+str_mon
            before2=str(int(year-2))+str_mon
            before1=str(int(year-1))+str_mon
            series=pd.DataFrame(np.array([float(data[before2]),float(data[before1]),float(data[mid]),float(data[after1])]))
            y=series.fillna(series.mean())
            tem_trend,r=np.polyfit(x,np.nan_to_num(y),1)
            box.append(tem_trend)
    return box


tem_tr=[]
for i in range(1,13):
    mon=month_trend(wf_nwf_8_10_night,i)
    tem_tr.append(mon)
column=np.arange(1,13,1)
tem_trend8_10_night=pd.DataFrame(np.array(tem_tr).reshape(12,419).T,columns=column)

#tem_trend8_10_night_test.to_csv('/mnt/e/US/data/result/lst_effect_night_p_value.csv')

data=pd.read_csv('/mnt/e/US/data/data_whole_add_ndvipeak.csv')
real=data[data['wf_end_builit_year0.95']<2018][data['wf_end_builit_year0.95']>0]

tem_tr=[]
for i in range(1,13):
    mon=month_trend(wf_nwf_6_8_night,i)
    tem_tr.append(mon)
column=np.arange(1,13,1)
tem_trend6_8_night=pd.DataFrame(np.array(tem_tr).reshape(12,419).T,columns=column)

tem_tr=[]
for i in range(1,13):
    mon=month_trend(wf_nwf_4_6_night,i)
    tem_tr.append(mon)
column=np.arange(1,13,1)
tem_trend4_6_night=pd.DataFrame(np.array(tem_tr).reshape(12,419).T,columns=column)

tem_tr=[]
for i in range(1,13):
    mon=month_trend(wf_nwf_2_4_night,i)
    tem_tr.append(mon)
column=np.arange(1,13,1)
tem_trend2_4_night=pd.DataFrame(np.array(tem_tr).reshape(12,419).T,columns=column)

tem_tr=[]
for i in range(1,13):
    mon=month_trend(wf_nwf_2_4_day,i)
    tem_tr.append(mon)
column=np.arange(1,13,1)
tem_trend2_4_day=pd.DataFrame(np.array(tem_tr).reshape(12,419).T,columns=column)

tem_tr=[]
for i in range(1,13):
    mon=month_trend(wf_nwf_4_6_day,i)
    tem_tr.append(mon)
column=np.arange(1,13,1)
tem_trend4_6_day=pd.DataFrame(np.array(tem_tr).reshape(12,419).T,columns=column)

tem_tr=[]
for i in range(1,13):
    mon=month_trend(wf_nwf_6_8_day,i)
    tem_tr.append(mon)
column=np.arange(1,13,1)
tem_trend6_8_day=pd.DataFrame(np.array(tem_tr).reshape(12,419).T,columns=column)

tem_tr=[]
for i in range(1,13):
    mon=month_trend(wf_nwf_8_10_day,i)
    tem_tr.append(mon)
column=np.arange(1,13,1)
tem_trend8_10_day=pd.DataFrame(np.array(tem_tr).reshape(12,419).T,columns=column)

columns_2_4_night=['lst_trend_night_jan_buff2_4','lst_trend_night_feb_buff2_4','lst_trend_night_mar_buff2_4','lst_trend_night_apr_buff2_4','lst_trend_night_may_buff2_4','lst_trend_night_jun_buff2_4'
                  ,'lst_trend_night_jul_buff2_4','lst_trend_night_aug_buff2_4','lst_trend_night_sep_buff2_4','lst_trend_night_oct_buff2_4','lst_trend_night_nov_buff2_4','lst_trend_night_dec_buff2_4']
columns_4_6_night=['lst_trend_night_jan_buff4_6','lst_trend_night_feb_buff4_6','lst_trend_night_mar_buff4_6','lst_trend_night_apr_buff4_6','lst_trend_night_may_buff4_6','lst_trend_night_jun_buff4_6'
                  ,'lst_trend_night_jul_buff4_6','lst_trend_night_aug_buff4_6','lst_trend_night_sep_buff4_6','lst_trend_night_oct_buff4_6','lst_trend_night_nov_buff4_6','lst_trend_night_dec_buff4_6']
columns_6_8_night=['lst_trend_night_jan_buff6_8','lst_trend_night_feb_buff6_8','lst_trend_night_mar_buff6_8','lst_trend_night_apr_buff6_8','lst_trend_night_may_buff6_8','lst_trend_night_jun_buff6_8'
                  ,'lst_trend_night_jul_buff6_8','lst_trend_night_aug_buff6_8','lst_trend_night_sep_buff6_8','lst_trend_night_oct_buff6_8','lst_trend_night_nov_buff6_8','lst_trend_night_dec_buff6_8']
columns_8_10_night=['lst_trend_night_jan_buff8_10','lst_trend_night_feb_buff8_10','lst_trend_night_mar_buff8_10','lst_trend_night_apr_buff8_10','lst_trend_night_may_buff8_10','lst_trend_night_jun_buff8_10'
                  ,'lst_trend_night_jul_buff8_10','lst_trend_night_aug_buff8_10','lst_trend_night_sep_buff8_10','lst_trend_night_oct_buff8_10','lst_trend_night_nov_buff8_10','lst_trend_night_dec_buff8_10']

columns_2_4_day=['lst_trend_day_jan_buff2_4','lst_trend_day_feb_buff2_4','lst_trend_day_mar_buff2_4','lst_trend_day_apr_buff2_4','lst_trend_day_may_buff2_4','lst_trend_day_jun_buff2_4'
                  ,'lst_trend_day_jul_buff2_4','lst_trend_day_aug_buff2_4','lst_trend_day_sep_buff2_4','lst_trend_day_oct_buff2_4','lst_trend_day_nov_buff2_4','lst_trend_day_dec_buff2_4']
columns_4_6_day=['lst_trend_day_jan_buff4_6','lst_trend_day_feb_buff4_6','lst_trend_day_mar_buff4_6','lst_trend_day_apr_buff4_6','lst_trend_day_may_buff4_6','lst_trend_day_jun_buff4_6'
                  ,'lst_trend_day_jul_buff4_6','lst_trend_day_aug_buff4_6','lst_trend_day_sep_buff4_6','lst_trend_day_oct_buff4_6','lst_trend_day_nov_buff4_6','lst_trend_day_dec_buff4_6']
columns_6_8_day=['lst_trend_day_jan_buff6_8','lst_trend_day_feb_buff6_8','lst_trend_day_mar_buff6_8','lst_trend_day_apr_buff6_8','lst_trend_day_may_buff6_8','lst_trend_day_jun_buff6_8'
                  ,'lst_trend_day_jul_buff6_8','lst_trend_day_aug_buff6_8','lst_trend_day_sep_buff6_8','lst_trend_day_oct_buff6_8','lst_trend_day_nov_buff6_8','lst_trend_day_dec_buff6_8']
columns_8_10_day=['lst_trend_day_jan_buff8_10','lst_trend_day_feb_buff8_10','lst_trend_day_mar_buff8_10','lst_trend_day_apr_buff8_10','lst_trend_day_may_buff8_10','lst_trend_day_jun_buff8_10'
                  ,'lst_trend_day_jul_buff8_10','lst_trend_day_aug_buff8_10','lst_trend_day_sep_buff8_10','lst_trend_day_oct_buff8_10','lst_trend_day_nov_buff8_10','lst_trend_day_dec_buff8_10']

tem_trend2_4_night.columns=columns_2_4_night
tem_trend4_6_night.columns=columns_4_6_night
tem_trend6_8_night.columns=columns_6_8_night
tem_trend8_10_night.columns=columns_8_10_night

tem_trend2_4_day.columns=columns_2_4_day
tem_trend4_6_day.columns=columns_4_6_day
tem_trend6_8_day.columns=columns_6_8_day
tem_trend8_10_day.columns=columns_8_10_day


tem_trend_buff=pd.concat([tem_trend2_4_day,tem_trend2_4_night,tem_trend4_6_day,tem_trend4_6_night,
                          tem_trend6_8_day,tem_trend6_8_night,tem_trend8_10_day,tem_trend8_10_night
                         ],axis=1)

