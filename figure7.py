#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import pandas as pd
import numpy as np
import math
from datetime import datetime
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
import seaborn as sns 
import xarray as xr
import matplotlib as mb
import matplotlib.pyplot as plt
import scipy.stats as stats0
import cartopy.io.shapereader as shpreader
import matplotlib.patches as mpatches
import shapely.geometry as sgeom

data=pd.read_csv('..//data/data_whole_add_ndvipeak.csv')
land_cover_nwf=pd.read_csv('../data/real_add_season_effect_add_nwf_land_cover.csv')
land=land_cover_nwf[['mode','land_diff']]
real=data[data['wf_end_builit_year0.95']<2018][data['wf_end_builit_year0.95']>0]
land.index=real.index
real=pd.concat([real,land],axis=1)

#8-10
night_jja=real.iloc[:,173:176].mean(1)*5
night_djf=pd.concat([real.iloc[:,179:180],real.iloc[:,168:170]],axis=1).mean(1)*5
night_mam=real.iloc[:,170:173].mean(1)*5
night_son=real.iloc[:,176:179].mean(1)*5

day_jja=real.iloc[:,161:164].mean(1)*5
day_djf=pd.concat([real.iloc[:,167:168].mean(1),real.iloc[:,156:158]],axis=1).mean(1)*5
day_mam=real.iloc[:,158:161].mean(1)*5
day_son=real.iloc[:,164:167].mean(1)*5

#6-8
night_jja_6_8=real.iloc[:,149:152].mean(1)*5
night_djf_6_8=pd.concat([real.iloc[:,155:156],real.iloc[:,144:146]],axis=1).mean(1)*5
night_mam_6_8=real.iloc[:,146:149].mean(1)*5
night_son_6_8=real.iloc[:,152:155].mean(1)*5

day_jja_6_8=real.iloc[:,137:140].mean(1)*5
day_djf_6_8=pd.concat([real.iloc[:,143:144].mean(1),real.iloc[:,132:134]],axis=1).mean(1)*5
day_mam_6_8=real.iloc[:,134:137].mean(1)*5
day_son_6_8=real.iloc[:,140:143].mean(1)*5


#4-6
night_jja_4_6=real.iloc[:,125:128].mean(1)*5
night_djf_4_6=pd.concat([real.iloc[:,131:132],real.iloc[:,120:122]],axis=1).mean(1)*5
night_mam_4_6=real.iloc[:,122:125].mean(1)*5
night_son_4_6=real.iloc[:,128:131].mean(1)*5

day_jja_4_6=real.iloc[:,113:116].mean(1)*5
day_djf_4_6=pd.concat([real.iloc[:,119:120].mean(1),real.iloc[:,108:110]],axis=1).mean(1)*5
day_mam_4_6=real.iloc[:,120:123].mean(1)*5
day_son_4_6=real.iloc[:,116:119].mean(1)*5


#2-4
night_jja_2_4=real.iloc[:,101:104].mean(1)*5
night_djf_2_4=pd.concat([real.iloc[:,107:108],real.iloc[:,96:98]],axis=1).mean(1)*5
night_mam_2_4=real.iloc[:,108:111].mean(1)*5
night_son_2_4=real.iloc[:,104:107].mean(1)*5

day_jja_2_4=real.iloc[:,89:92].mean(1)*5
day_djf_2_4=pd.concat([real.iloc[:,95:96].mean(1),real.iloc[:,84:86]],axis=1).mean(1)*5
day_mam_2_4=real.iloc[:,96:99].mean(1)*5
day_son_2_4=real.iloc[:,92:95].mean(1)*5


ndvi_peak=real['ndvi_peak_trend_8_10']*10000*5
real_add_season_effect=pd.concat([real,
                                  pd.DataFrame(day_mam,columns=['lst_effect_day_mam']),
                                  pd.DataFrame(day_jja,columns=['lst_effect_day_jja']),
                                  pd.DataFrame(day_son,columns=['lst_effect_day_son']),
                                  pd.DataFrame(day_djf,columns=['lst_effect_day_djf']),
                                  pd.DataFrame(night_mam,columns=['lst_effect_night_mam']),
                                  pd.DataFrame(night_jja,columns=['lst_effect_night_jja']),
                                  pd.DataFrame(night_son,columns=['lst_effect_night_son']),
                                  pd.DataFrame(night_djf,columns=['lst_effect_night_djf']),
                                  
                                  pd.DataFrame(day_mam_6_8,columns=['lst_effect_day_mam_6_8']),
                                  pd.DataFrame(day_jja_6_8,columns=['lst_effect_day_jja_6_8']),
                                  pd.DataFrame(day_son_6_8,columns=['lst_effect_day_son_6_8']),
                                  pd.DataFrame(day_djf_6_8,columns=['lst_effect_day_djf_6_8']),
                                  pd.DataFrame(night_mam_6_8,columns=['lst_effect_night_mam_6_8']),
                                  pd.DataFrame(night_jja_6_8,columns=['lst_effect_night_jja_6_8']),
                                  pd.DataFrame(night_son_6_8,columns=['lst_effect_night_son_6_8']),
                                  pd.DataFrame(night_djf_6_8,columns=['lst_effect_night_djf_6_8']),
                                  
                                  pd.DataFrame(day_mam_4_6,columns=['lst_effect_day_mam_4_6']),
                                  pd.DataFrame(day_jja_4_6,columns=['lst_effect_day_jja_4_6']),
                                  pd.DataFrame(day_son_4_6,columns=['lst_effect_day_son_4_6']),
                                  pd.DataFrame(day_djf_4_6,columns=['lst_effect_day_djf_4_6']),
                                  pd.DataFrame(night_mam_4_6,columns=['lst_effect_night_mam_4_6']),
                                  pd.DataFrame(night_jja_4_6,columns=['lst_effect_night_jja_4_6']),
                                  pd.DataFrame(night_son_4_6,columns=['lst_effect_night_son_4_6']),
                                  pd.DataFrame(night_djf_4_6,columns=['lst_effect_night_djf_4_6']),
                                  
                                  pd.DataFrame(day_mam_2_4,columns=['lst_effect_day_mam_2_4']),
                                  pd.DataFrame(day_jja_2_4,columns=['lst_effect_day_jja_2_4']),
                                  pd.DataFrame(day_son_2_4,columns=['lst_effect_day_son_2_4']),
                                  pd.DataFrame(day_djf_2_4,columns=['lst_effect_day_djf_2_4']),
                                  pd.DataFrame(night_mam_2_4,columns=['lst_effect_night_mam_2_4']),
                                  pd.DataFrame(night_jja_2_4,columns=['lst_effect_night_jja_2_4']),
                                  pd.DataFrame(night_son_2_4,columns=['lst_effect_night_son_2_4']),
                                  pd.DataFrame(night_djf_2_4,columns=['lst_effect_night_djf_2_4']),
                                  
                                  pd.DataFrame(ndvi_peak,columns=['ndvi_peak_effect'])
                                 ],axis=1)

#divide wind farms into small,medium,large
tnum_small=real_add_season_effect[real_add_season_effect['p_tnum']<=25]                               # 108
tnum_medium=real_add_season_effect[real_add_season_effect['p_tnum']>25][real['p_tnum']<=75]           # 106 
tnum_large=real_add_season_effect[real_add_season_effect['p_tnum']>75]                                # 105

#划分下垫面类型(三种主要的280/320)
cover_forest=real_add_season_effect[real['land_cover']==4.0]                      #38
cover_grass=real_add_season_effect[real['land_cover']==10.0]                       #127
cover_crop=real_add_season_effect[real['land_cover']==12.0]                      #115
cover_other=real_add_season_effect[real['land_cover']!=12.0][real['land_cover']!=10.0][real['land_cover']!=4.0]  #39

wind_str=['wind_100m_mam_night','wind_100m_jja_night','wind_100m_son_night','wind_100m_djf_night']
lst_str=['lst_effect_night_mam','lst_effect_night_jja','lst_effect_night_son','lst_effect_night_djf']

mon_var=['lst_trend_night_jan_buff8_10','lst_trend_night_feb_buff8_10','lst_trend_night_mar_buff8_10','lst_trend_night_apr_buff8_10',
    'lst_trend_night_may_buff8_10','lst_trend_night_jun_buff8_10','lst_trend_night_jul_buff8_10','lst_trend_night_aug_buff8_10',
    'lst_trend_night_sep_buff8_10','lst_trend_night_oct_buff8_10','lst_trend_night_nov_buff8_10','lst_trend_night_dec_buff8_10']

crop_wind=cover_crop[wind_str].mean()
crop_lst=cover_crop[lst_str].mean()

forest_wind=cover_forest[wind_str].mean()
forest_lst=cover_forest[lst_str].mean()

grass_wind=cover_grass[wind_str].mean()
grass_lst=cover_grass[lst_str].mean()
#plt.plot(a,grass_lst)
plt.plot(a,grass_wind)

#divide wind farms into three major land cover types(三种主要的280/319)
#nwf
cover_forest_nwf=real_add_season_effect[real['mode']==4.0]                      #38
cover_grass_nwf=real_add_season_effect[real['mode']==10.0]                       #127
cover_crop_nwf=real_add_season_effect[real['mode']==12.0]                      #115
cover_other_nwf=real_add_season_effect[real['mode']!=12.0][real['mode']!=10.0][real['mode']!=4.0]  #39
ndvi_var=['ndvi_peak_trend_2_4','ndvi_peak_trend_4_6','ndvi_peak_trend_6_8','ndvi_peak_trend_8_10']
buffer=['0','2_4','4_6','6_8','8_10']

ndvi=real[['ndvi_peak_trend_2_4','ndvi_peak_trend_4_6','ndvi_peak_trend_6_8','ndvi_peak_trend_8_10']]*10000*10
ndvi_small=tnum_small[ndvi_var]*10000*5
ndvi_medium=tnum_medium[ndvi_var]*10000*5
ndvi_large=tnum_large[ndvi_var]*10000*5

def to_ndvi_decline(data):
    
    data['0_2']=data['ndvi_peak_trend_8_10']
    data['2_4']=data['ndvi_peak_trend_8_10']-data['ndvi_peak_trend_2_4']
    data['4_6']=data['ndvi_peak_trend_8_10']-data['ndvi_peak_trend_4_6']
    data['6_8']=data['ndvi_peak_trend_8_10']-data['ndvi_peak_trend_6_8']
    data['8_10']=data['ndvi_peak_trend_8_10']-data['ndvi_peak_trend_8_10']
    return data[['0_2','2_4','4_6','6_8','8_10']]

region=['WF','NWF2_4','NWF4_6','NWF6_8','NWF8_10']


lst_day_djf_var=['lst_effect_day_djf_2_4','lst_effect_day_djf_4_6','lst_effect_day_djf_6_8','lst_effect_day_djf']
lst_night_djf_var=['lst_effect_night_djf_2_4','lst_effect_night_djf_4_6','lst_effect_night_djf_6_8','lst_effect_night_djf']

lst_day_jja_var=['lst_effect_day_jja_2_4','lst_effect_day_jja_4_6','lst_effect_day_jja_6_8','lst_effect_day_jja']
lst_night_jja_var=['lst_effect_night_jja_2_4','lst_effect_night_jja_4_6','lst_effect_night_jja_6_8','lst_effect_night_jja']

lst_day_son_var=['lst_effect_day_son_2_4','lst_effect_day_son_4_6','lst_effect_day_son_6_8','lst_effect_day_son']
lst_night_son_var=['lst_effect_night_son_2_4','lst_effect_night_son_4_6','lst_effect_night_son_6_8','lst_effect_night_son']

lst_day_mam_var=['lst_effect_day_mam_2_4','lst_effect_day_mam_4_6','lst_effect_day_mam_6_8','lst_effect_day_mam']
lst_night_mam_var=['lst_effect_night_mam_2_4','lst_effect_night_mam_4_6','lst_effect_night_mam_6_8','lst_effect_night_mam']

def to_lst_decline(data,strs):
    
    data['0_2']=data[strs]
    data['2_4']=data[strs]-data[strs+'_2_4']
    data['4_6']=data[strs]-data[strs+'_4_6']
    data['6_8']=data[strs]-data[strs+'_6_8']
    data['8_10']=data[strs]-data[strs]
    return data[['0_2','2_4','4_6','6_8','8_10']]

def plot_cover(var,strs):
    
    lst_all=real_add_season_effect[var]
    lst_small=cover_crop[var]
    lst_medium=cover_forest[var]
    lst_large=cover_grass[var]
    lst_big=cover_other[var]

#    plt.figure(figsize=(10,6))
    plt.plot(buffer,to_lst_decline(lst_all,strs).mean()*10,label='All')
    plt.plot(buffer,to_lst_decline(lst_small,strs).mean()*10,label='crop')
    plt.plot(buffer,to_lst_decline(lst_medium,strs).mean()*10,label='forest')
    plt.plot(buffer,to_lst_decline(lst_large,strs).mean()*10,label='grass')
    plt.plot(buffer,to_lst_decline(lst_big,strs).mean()*10,label='other')

def plot_scale(var,strs,ax):
    
    lst_all=real_add_season_effect[var]
    lst_small=tnum_small[var]
    lst_medium=tnum_medium[var]
    lst_large=tnum_large[var]

#    plt.figure(figsize=(10,6))
    ax.errorbar(region,to_lst_decline(lst_all,strs).mean(),label='All',c='k',linewidth=3.5,linestyle='--')
    ax.errorbar(region,to_lst_decline(lst_small,strs).mean(),yerr=np.std(to_lst_decline(lst_small,strs))/20,label='Small',c='Green',marker='o',markersize=3,linewidth=2.5)
    ax.errorbar(region,to_lst_decline(lst_medium,strs).mean(),yerr=np.std(to_lst_decline(lst_medium,strs))/20,label='Medium',c='Blue',marker='o',markersize=5,linewidth=2.5)
    ax.errorbar(region,to_lst_decline(lst_large,strs).mean(),yerr=np.std(to_lst_decline(lst_large,strs))/20,label='Large',c='Red',marker='o',markersize=8,linewidth=2.5)
    ax.hlines(0,-0.5,4.5,linestyles=(0, (2, 2)),linewidth=2,colors='grey')
    
    
    ax.tick_params(direction='in')
#    ax.set_xticklabels(fontsize=24)
#   ax.set_yticklabels(fontsize=24)
#    plset_t.xlabel('Distance from WFM',fontsize=20)
#    plt.ylabel('NDVI effect',fontsize=20)
#    plt.legend(loc='lower right',framealpha=0,fontsize=16)
#    plt.title('Different Scale',fontsize=20)

def plot_scale_wo_err(var,strs,ax):
    
    lst_all=real_add_season_effect[var]
    lst_small=tnum_small[var]
    lst_medium=tnum_medium[var]
    lst_large=tnum_large[var]

#    plt.figure(figsize=(10,6))
    ax.errorbar(region,to_lst_decline(lst_all,strs).mean()/2,label='All',c='k',linewidth=3.5,linestyle='--')
    ax.errorbar(region,to_lst_decline(lst_small,strs).mean()/2,label='Small',c='Green',marker='o',markersize=3,linewidth=2.5)
    ax.errorbar(region,to_lst_decline(lst_medium,strs).mean()/2,label='Medium',c='Blue',marker='o',markersize=5,linewidth=2.5)
    ax.errorbar(region,to_lst_decline(lst_large,strs).mean()/2,label='Large',c='Red',marker='o',markersize=8,linewidth=2.5)
    ax.hlines(0,-0.5,4.5,linestyles=(0, (2, 2)),linewidth=2,colors='grey')
    
    
    ax.tick_params(direction='in')
#    ax.set_xticklabels(fontsize=24)
#   ax.set_yticklabels(fontsize=24)
#    plset_t.xlabel('Distance from WFM',fontsize=20)
#    plt.ylabel('NDVI effect',fontsize=20)
#    plt.legend(loc='lower right',framealpha=0,fontsize=16)
#    plt.title('Different Scale',fontsize=20)
var=['lst_trend_night_jan_buff8_10','lst_trend_night_feb_buff8_10','lst_trend_night_mar_buff8_10',
    'lst_trend_night_apr_buff8_10','lst_trend_night_may_buff8_10','lst_trend_night_jun_buff8_10',
    'lst_trend_night_jul_buff8_10','lst_trend_night_aug_buff8_10','lst_trend_night_sep_buff8_10',
    'lst_trend_night_oct_buff8_10','lst_trend_night_nov_buff8_10','lst_trend_night_dec_buff8_10']
mon=np.arange(1,13,1)
var_8_10=['lst_effect_night_mam','lst_effect_night_jja',
         'lst_effect_night_son','lst_effect_night_djf']
season=['MAM','JJA','SON','DJF']

gath=pd.DataFrame(np.array([cover_grass[var_8_10].mean(),cover_crop[var_8_10].mean(),
         cover_forest[var_8_10].mean(),cover_other[var_8_10].mean(),
        real_add_season_effect[var_8_10].mean()]).T, columns=['Grass','Crop','Forest','Others','All'])
gath.index=['MAM','JJA','SON','DJF']

cover_grass[var_8_10].quantile(0.5)

grass_error=np.array(cover_grass[var_8_10].quantile(0.55))-cover_grass[var_8_10].quantile(0.5)
crop_error=np.array(cover_crop[var_8_10].quantile(0.55))-cover_crop[var_8_10].quantile(0.5)
forest_error=np.array(cover_forest[var_8_10].quantile(0.55))-cover_forest[var_8_10].quantile(0.5)
other_error=np.array(cover_other[var_8_10].quantile(0.55))-cover_other[var_8_10].quantile(0.5)
all_error=np.array(real_add_season_effect[var_8_10].quantile(0.55))-real_add_season_effect[var_8_10].quantile(0.5)

#divide wind farm size in land cover type
#   forest
forest_small=cover_forest[cover_forest['p_tnum']<=25]                               # 108
forest_medium=cover_forest[cover_forest['p_tnum']>25][cover_forest['p_tnum']<=75]           # 106 
forest_large=cover_forest[cover_forest['p_tnum']>75]                                # 105

#   grass
grass_small=cover_grass[cover_grass['p_tnum']<=25]                               # 108
grass_medium=cover_grass[cover_grass['p_tnum']>25][cover_grass['p_tnum']<=75]           # 106 
grass_large=cover_grass[cover_grass['p_tnum']>75]                                # 105

#   crop
crop_small=cover_crop[cover_crop['p_tnum']<=25]                               # 108
crop_medium=cover_crop[cover_crop['p_tnum']>25][cover_crop['p_tnum']<=75]           # 106 
crop_large=cover_crop[cover_crop['p_tnum']>75]                                # 105

#  other
other_small=cover_other[cover_other['p_tnum']<=25]                               # 108
other_medium=cover_other[cover_other['p_tnum']>25][cover_other['p_tnum']<=75]           # 106 
other_large=cover_other[cover_other['p_tnum']>75]                                # 105

forest_s = np.array([forest_small['lst_effect_night_mam'].mean(),forest_medium['lst_effect_night_mam'].mean(),forest_large['lst_effect_night_mam'].mean(),
                    forest_small['lst_effect_night_jja'].mean(),forest_medium['lst_effect_night_jja'].mean(),forest_large['lst_effect_night_jja'].mean(),
                    forest_small['lst_effect_night_son'].mean(),forest_medium['lst_effect_night_son'].mean(),forest_large['lst_effect_night_son'].mean(),
                    forest_small['lst_effect_night_djf'].mean(),forest_medium['lst_effect_night_djf'].mean(),forest_large['lst_effect_night_djf'].mean()])
forest_s=forest_s.reshape(4,3)


crop_s = np.array([crop_small['lst_effect_night_mam'].mean(),crop_medium['lst_effect_night_mam'].mean(),crop_large['lst_effect_night_mam'].mean(),
                    crop_small['lst_effect_night_jja'].mean(),crop_medium['lst_effect_night_jja'].mean(),crop_large['lst_effect_night_jja'].mean(),
                    crop_small['lst_effect_night_son'].mean(),crop_medium['lst_effect_night_son'].mean(),crop_large['lst_effect_night_son'].mean(),
                    crop_small['lst_effect_night_djf'].mean(),crop_medium['lst_effect_night_djf'].mean(),crop_large['lst_effect_night_djf'].mean()])
crop_s=crop_s.reshape(4,3)

grass_s = np.array([grass_small['lst_effect_night_mam'].mean(),grass_medium['lst_effect_night_mam'].mean(),grass_large['lst_effect_night_mam'].mean(),
                    grass_small['lst_effect_night_jja'].mean(),grass_medium['lst_effect_night_jja'].mean(),grass_large['lst_effect_night_jja'].mean(),
                    grass_small['lst_effect_night_son'].mean(),grass_medium['lst_effect_night_son'].mean(),grass_large['lst_effect_night_son'].mean(),
                    grass_small['lst_effect_night_djf'].mean(),grass_medium['lst_effect_night_djf'].mean(),grass_large['lst_effect_night_djf'].mean()])
grass_s=grass_s.reshape(4,3)

other_s = np.array([other_small['lst_effect_night_mam'].mean(),other_medium['lst_effect_night_mam'].mean(),other_large['lst_effect_night_mam'].mean(),
                    other_small['lst_effect_night_jja'].mean(),other_medium['lst_effect_night_jja'].mean(),other_large['lst_effect_night_jja'].mean(),
                    other_small['lst_effect_night_son'].mean(),other_medium['lst_effect_night_son'].mean(),other_large['lst_effect_night_son'].mean(),
                    other_small['lst_effect_night_djf'].mean(),other_medium['lst_effect_night_djf'].mean(),other_large['lst_effect_night_djf'].mean()])
other_s=other_s.reshape(4,3)

fig, ax= plt.subplots(figsize=(12,8))

plt.rcParams['font.family'] = "Times New Roman"
x = np.arange(4)
width = .23
# fig.figsize=(12,8)

ax.bar(x - 2*width+0.2, np.array(cover_grass[var_8_10].mean())/2, width,label='Grassland',color='#EEB422',ec='black',lw=.5,zorder=3)
grass = np.array(cover_grass[var_8_10].mean())
wi = x - 2*width+0.2
for i in range(0,4):
    xl = wi[i]-0.1
    yl = grass[i]/2+0.01
    ax.text(xl, yl, str(round(grass[i]/2,2)), color='#EEB422', fontweight='bold',fontsize=12)

    
    
ax.bar(x - width+0.2, np.array(cover_crop[var_8_10].mean())/2, width, label='Cropland',color='#CB181B',ec='black',lw=.5,zorder=3)
crop = np.array(cover_crop[var_8_10].mean())
wi1 = x - width+0.2
for i in range(0,4):
    xl = wi1[i]-0.08
    yl = crop[i]/2+0.01
    ax.text(xl, yl, str(round(crop[i]/2,2)), color='#CB181B', fontweight='bold',fontsize=12)
    
    
ax.bar(x+0.2  , np.array(cover_forest[var_8_10].mean())/2, width, label='Forest',color='#008B45',ec='black',lw=.5,zorder=3)
forest = np.array(cover_forest[var_8_10].mean())
wi2 = x+0.2
for i in range(0,4):
    xl1 = wi2[i]-0.07-0.02
    xl2 = wi2[i]-0.1-0.02
    yl = forest[i]/2+0.01
    if forest[i]>0:
        ax.text(xl1, yl, str(round(forest[i]/2,2)), color='#008B45', fontweight='bold',fontsize=12)
    else:
        ax.text(xl2, yl-0.04, str(round(forest[i]/2,2)), color='#008B45', fontweight='bold',fontsize=12)
    
    
ax.bar(x + width+0.2, np.array(cover_other[var_8_10].mean())/2,width, label='Others',color='#130074',ec='black',lw=.5,zorder=3)
other = np.array(cover_other[var_8_10].mean())
wi3 = x + width+0.2
for i in range(0,4):
    xl = wi3[i]-0.08
    yl = other[i]/2+0.01
    if other[i]>0:
        ax.text(xl, yl, str(round(other[i]/2,2)), color='#130074', fontweight='bold',fontsize=12)
    else:
        ax.text(xl, yl-0.04, str(round(other[i]/2,2)), color='#130074', fontweight='bold',fontsize=12)
    
    
# ax.bar(x + width*2+0.04, np.array(real_add_season_effect[var_8_10].mean()),width, label='All',color='grey',zorder=3)
# al = np.array(real_add_season_effect[var_8_10].mean())
# wi4 = x + width*2+0.04
# for i in range(0,4):
#     xl = wi4[i]-0.08
#     yl = al[i]+0.01
#     if al[i]>0:
#         ax.text(xl, yl, str(round(al[i],2)), color='grey', fontweight='bold',fontsize=12)
#     else:
#         ax.text(xl, yl-0.04, str(round(al[i],2)), color='grey', fontweight='bold',fontsize=12)


ax.hlines(0.11,-0.8,4,linestyles='--',linewidth=2,colors='#EEB422')
ax.hlines(0.17,-0.8,4,linestyles='--',linewidth=2,colors='#CB181B')
ax.hlines(-0.01,-0.8,4,linestyles='--',linewidth=2,colors='#008B45')
ax.hlines(-0.06,-0.8,4,linestyles='--',linewidth=2,colors='#130074')

ax.text(3.7,0.08,str(round(cover_grass[var_8_10].mean().mean()/2,2)),color='#EEB422', fontweight='bold',fontsize=14)
ax.text(3.7,0.14,str(round(cover_crop[var_8_10].mean().mean()/2,2)),color='#CB181B', fontweight='bold',fontsize=14)
ax.text(3.68,-0.04,str(round(cover_forest[var_8_10].mean().mean()/2,2)),color='#008B45', fontweight='bold',fontsize=14)
ax.text(3.68,-0.09,str(round(cover_other[var_8_10].mean().mean()/2,2)),color='#130074', fontweight='bold',fontsize=14)
 
ax.set_ylabel(r'$\Delta$LST ($^\circ$C)',fontsize=24)
ax.set_xlabel('Seasons',fontsize=24)
ax.set_ylim([-0.35,0.35])
ax.set_yticks([-0.3,-0.2,-0.1,0,0.1,0.2,0.3])
ax.set_yticklabels([-0.3,-0.2,-0.1,0,0.1,0.2,0.3],fontsize=20)
ax.hlines(0,-0.6,4,linestyles='-',linewidth=1,colors='k')
ax.set_xlim(-0.6,4)
ax.set_xticks([0,1,2,3])
ax.set_xticklabels(['Spring','Summer','Autumn','Winter'],fontsize=20)

# for spine in ['top','right']:
#     ax.spines[spine].set_color('none')
# #    ax.legend(fontsize=7,frameon=False)
#     text_font = {'size':'18','weight':'bold','color':'black'}

plt.legend(loc='lower right',framealpha=0,fontsize=20)
#ax.grid(color='grey', linestyle='--', linewidth=1,axis='y',zorder=0)
#ax.grid(color='grey', linestyle='--', linewidth=1,axis='x',zorder=0)
plt.savefig('../results/figures/fig7 land cover on night3 .jpg',bbox_inches='tight',dpi=300)

