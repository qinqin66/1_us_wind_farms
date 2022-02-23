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

[to_ndvi_decline(cover_forest[ndvi_var]).mean()*10000*10,
 to_ndvi_decline(cover_grass[ndvi_var]).mean()*10000*10,
 to_ndvi_decline(cover_crop[ndvi_var]).mean()*10000*10,
 to_ndvi_decline(cover_other[ndvi_var]).mean()*10000*10]

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

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False,figsize=(12,8))
fig.subplots_adjust(wspace=0.1,hspace=0.18)
plot_scale_wo_err(lst_night_mam_var,'lst_effect_night_mam',ax1)
ax1.vlines(0,-0.3,0.25,linestyles=(0, (2, 2)),linewidth=2,colors='k')
ax1.text(0, 1.02, 'a',transform=ax1.transAxes,fontsize=20, fontweight='bold')
ax1.set_title('Spring',fontsize=20)
ax1.set_ylim(-0.3,0.25)
ax1.set_xlim(-0.5,4.5)
ax1.set_yticks([-0.3,-0.2,-0.1,0,0.1,0.2])
ax1.set_yticklabels([-0.3,-0.2,-0.1,0,0.1,0.2],fontsize=12)
ax1.legend(loc='lower right',framealpha=0,fontsize=12)
ax1.set_ylabel(r'$\Delta$LST($^\circ$C)',position=(0,-0.1),fontsize=22)
ax1.text(-0.48,0.13,'0.13',c='r',fontsize=12)
ax1.text(-0.48,-0.05,'-0.05',c='Blue',fontsize=12)
ax1.text(-0.48,-0.13,'-0.14',c='Green',fontsize=12)
ax1.tick_params(direction='out')


plot_scale_wo_err(lst_night_jja_var,'lst_effect_night_jja',ax2)
ax2.vlines(0,-0.3,0.25,linestyles=(0, (2, 2)),linewidth=2,colors='k')
ax2.set_ylim(-0.3,0.25)
ax2.set_xlim(-0.5,4.5)
ax2.set_yticks([-0.3,-0.2,-0.1,0,0.1,0.2])
ax2.set_yticklabels([-0.3,-0.2,-0.1,0,0.1,0.2],fontsize=12)
ax2.set_yticklabels('')
ax2.text(0, 1.02, 'b',transform=ax2.transAxes,fontsize=20, fontweight='bold')
ax2.set_title('Summer',fontsize=20)
ax2.text(-0.48,0.2,'0.18',c='r',fontsize=12)
ax2.text(-0.48,0.1,'0.12',c='Blue',fontsize=12)
ax2.text(-0.48,-0.15,'-0.10',c='Green',fontsize=12)
ax2.tick_params(direction='out')




plot_scale_wo_err(lst_night_son_var,'lst_effect_night_son',ax3)
ax3.set_ylim(-0.05,0.3)
ax3.set_xlim(-0.5,4.5)
ax3.set_yticklabels([-0.05,0,0.05,'0.10',0.15,'0.20',0.25],fontsize=12)
ax3.vlines(0,-0.05,0.3,linestyles=(0, (2, 2)),linewidth=2,colors='k')

ax3.text(0, 1.02, 'c',transform=ax3.transAxes,fontsize=20, fontweight='bold')
ax3.set_title('Autumn',fontsize=20)
ax3.set_xticklabels(region,fontsize=12)
ax3.text(-0.48,0.2,'0.18',c='r',fontsize=12)
ax3.text(-0.48,0.1,'0.14',c='Blue',fontsize=12)
ax3.text(-0.48,0.15,'0.16',c='Green',fontsize=12)
ax3.tick_params(direction='out')


plot_scale_wo_err(lst_night_djf_var,'lst_effect_night_djf',ax4)
ax4.set_ylim(-0.05,0.3)
ax4.set_xlim(-0.5,4.5)
ax4.set_yticklabels('')
ax4.vlines(0,-0.05,0.3,linestyles=(0, (2, 2)),linewidth=2,colors='k')
ax4.text(0, 1.02, 'd',transform=ax4.transAxes,fontsize=20, fontweight='bold')
ax4.set_title('Winter',fontsize=20)
ax4.set_xticklabels(region,fontsize=12)
ax4.text(-0.48,0.2,'0.21',c='r',fontsize=12)
ax4.text(-0.48,0.15,'0.20',c='Blue',fontsize=12)
ax4.text(-0.48,0.1,'0.14',c='Green',fontsize=12)
ax4.tick_params(direction='out')
ax4.set_xlabel('Distance from wind farm region',position=(-0.1,-0.2),fontsize=22)

plt.savefig('../results/figures/fig5. LST effects of different size_without_errorbar.10.14.jpg',bbox_inches='tight',dpi=300)



    
