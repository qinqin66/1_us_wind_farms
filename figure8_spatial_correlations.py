#!/usr/bin/env python
# coding: utf-8

#gis3 env
import xarray as xr
import pandas as pd
import numpy as np
import math
from datetime import datetime
# import cartopy
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from cartopy.mpl.geoaxes import GeoAxes
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
import seaborn as sns 
import xarray as xr
import matplotlib as mb
import matplotlib.pyplot as plt
import scipy.stats as stats
# import cartopy.io.shapereader as shpreader
# import matplotlib.patches as mpatches
# import shapely.geometry as sgeom
from matplotlib.patches import Ellipse
import matplotlib as mpl

new_climate_var = pd.read_csv('../data/data_whole_update_climate_var.csv')[['wind_100m_annual_mean', 't2m_annual_max', 't2m_annual_min', 't2m_annual_mean', 'tp_annual_mean']]
real1=pd.read_csv('../data/real_add_season_effect_add_nwf_land_cover.csv')
real1 = real1.set_index(real1['Unnamed: 0.1'])
new_climate = pd.concat([pd.DataFrame([new_climate_var['wind_100m_annual_mean'][real1.index]]),
                            pd.DataFrame([new_climate_var['t2m_annual_max'][real1.index]]),
                            pd.DataFrame([new_climate_var['t2m_annual_min'][real1.index]]),
                            pd.DataFrame([new_climate_var['t2m_annual_mean'][real1.index]]),
                            pd.DataFrame([new_climate_var['tp_annual_mean'][real1.index]])],axis=0)
new_climate = pd.DataFrame(np.array(new_climate).reshape(320,5),columns=['wind_100m_annual_mean', 't2m_annual_max', 't2m_annual_min', 't2m_annual_mean', 'tp_annual_mean'])

new_climate.index = real1.index
real = pd.concat([real1,new_climate],axis=1)
day_annual=real.iloc[:,158:170].mean(1)*5
night_annual=real.iloc[:,170:182].mean(1)*5

real['annual_day']= day_annual
real['annual_night']= night_annual
#real.to_csv('../data/share/data_share.csv')
real['annual_day']= day_annual

night_mam_var=['lst_effect_night_mam','xlong','ylat','p_tnum','ndvi_95','elevation','ele_minus',
               't2m_annual_mean','lapse_mam_day','tp_annual_mean','wind_100m_annual_mean','ndvi_peak_trend_8_10']
night_jja_var=['lst_effect_night_jja','xlong','ylat','p_tnum','ndvi_95','elevation','ele_minus',
               't2m_annual_mean','lapse_jja_day','tp_annual_mean','wind_100m_annual_mean','ndvi_peak_trend_8_10']
night_son_var=['lst_effect_night_son','xlong','ylat','p_tnum','ndvi_95','elevation','ele_minus',
               't2m_annual_mean','lapse_son_day','tp_annual_mean','wind_100m_annual_mean','ndvi_peak_trend_8_10']
night_djf_var=['lst_effect_night_djf','xlong','ylat','p_tnum','ndvi_95','elevation','ele_minus',
               't2m_annual_mean','lapse_djf_day','tp_annual_mean','wind_100m_annual_mean','ndvi_peak_trend_8_10']

day_mam_var=['lst_effect_day_mam','xlong','ylat','p_tnum','ndvi_95','elevation','ele_minus',
               't2m_annual_mean','lapse_mam_night','tp_annual_mean','wind_100m_annual_mean','ndvi_peak_trend_8_10']
day_jja_var=['lst_effect_day_jja','xlong','ylat','p_tnum','ndvi_95','elevation','ele_minus',
               't2m_annual_mean','lapse_jja_night','tp_annual_mean','wind_100m_annual_mean','ndvi_peak_trend_8_10']
day_son_var=['lst_effect_day_son','xlong','ylat','p_tnum','ndvi_95','elevation','ele_minus',
               't2m_annual_mean','lapse_son_night','tp_annual_mean','wind_100m_annual_mean','ndvi_peak_trend_8_10']
day_djf_var=['lst_effect_day_djf','xlong','ylat','p_tnum','ndvi_95','elevation','ele_minus',
               't2m_annual_mean','lapse_djf_night','tp_annual_mean','wind_100m_annual_mean','ndvi_peak_trend_8_10']

day_var=['annual_day','xlong','ylat','p_tnum','ndvi_95','elevation','ele_minus',
               't2m_annual_mean','lapse_djf_day','tp_annual_mean','wind_100m_annual_mean','ndvi_peak_trend_8_10']
night_var=['annual_night','xlong','ylat','p_tnum','ndvi_95','elevation','ele_minus',
               't2m_annual_mean','lapse_djf_night','tp_annual_mean','wind_100m_annual_mean','ndvi_peak_trend_8_10']
ndvi_var=['ndvi_peak_trend_8_10','xlong','ylat','p_tnum','ndvi_95','elevation','ele_minus',
               't2m_annual_mean','lapse_djf_night','tp_annual_mean','wind_100m_annual_mean']
var_new=['Lon','Lat','Turb$_{num}$','NdVI$_{peak}$','Elevation',r'$\Delta$Elevation',
     'T$_{2m}$','Lapse','Precip','V$_{100m}$',r'$\Delta$NDVI']
season=['Annual','Spring','Summer',"Autumn","Winter"]
season2=["ANN",'MAM','JJA','SON',"DJF"]

stats.pearsonr(real['lst_effect_day_jja'].fillna(real['lst_effect_day_jja'].mean()),
               real['ndvi_peak_trend_8_10'].fillna(real['ndvi_peak_trend_8_10'].mean()))

mam_day=real[day_mam_var]
jja_day=real[day_jja_var]
son_day=real[day_son_var]
djf_day=real[day_djf_var]

mam_night=real[night_mam_var]
jja_night=real[night_jja_var]
son_night=real[night_son_var]
djf_night=real[night_djf_var]

annual_day = real[day_var]
annual_night = real[night_var]
ndvi= real[ndvi_var]

def cal_corr(data,variable):
    corr_box=[]
    for var in variable:
        corr=stats.pearsonr(data.iloc[:,0].fillna(data.iloc[:,0].mean()),data[var].fillna(data[var].mean()))
        corr_box.append(corr)
    return corr_box

data = jja_night
stats.pearsonr(data.iloc[:,0].fillna(data.iloc[:,0].mean()),data['ndvi_peak_trend_8_10'].fillna(data['ndvi_peak_trend_8_10'].mean()))

total_night=np.array([cal_corr(annual_night,night_var)[1:12],cal_corr(mam_night,night_mam_var)[1:12],cal_corr(jja_night,night_jja_var)[1:12],
                   cal_corr(son_night,night_son_var)[1:12],cal_corr(djf_night,night_djf_var)[1:12]])

total_day=np.array([cal_corr(annual_day,day_var)[1:12],cal_corr(mam_day,day_mam_var)[1:12],cal_corr(jja_day,day_jja_var)[1:12],
                   cal_corr(son_day,day_son_var)[1:12],cal_corr(djf_day,day_djf_var)[1:12]])

total_ndvi = np.array(cal_corr(ndvi,ndvi_var)[1:11])
total_ndvi=np.array(pd.DataFrame(total_ndvi).append(cal_corr(ndvi,ndvi_var)[0:1]))

gridspec = dict(hspace=0.0, width_ratios=[4, 4,1])
fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3,figsize=(12,8) ,gridspec_kw=gridspec)

im = ax1.imshow(total_day[:,:,0].T,vmin=-0.15,vmax=0.15,cmap='bwr',aspect='auto')
ax1.set_xticks(np.arange(len(season)))
ax1.set_yticks(np.arange(len(var_new)))
ax1.set_xticklabels(season2,fontsize=16,rotation=0)
ax1.set_yticklabels(var_new,fontsize=16,rotation=45)
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
for i in range(len(var_new)):
    for j in range(len(season)):
        if round(total_day[:,:,1].T[i, j])<0.05:
            te=str(('%.2f'%total_day[:,:,0].T[i, j]))+'*'
            text = ax1.text(j, i, te,
                ha="center", va="center", color="k",fontsize=15)
        else: 
            text= ax1.text(j, i, ('%.2f'%total_day[:,:,0].T[i, j]),
                           ha="center", va="center", color="k",fontsize=15)
                 
ax1.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
ax1.text(-0.1, 1.0, 'a', fontsize=20, transform=ax1.transAxes, fontweight='bold')
fig.tight_layout()
# cbar = ax1.figure.colorbar(im, ax=ax1,cmap="bwr")
ax1.set_xlabel('Daytime',fontsize=20)

im2 = ax2.imshow(total_night[:,:,0].T,vmin=-0.15,vmax=0.15,cmap='bwr',aspect='auto')
ax2.set_xticks(np.arange(len(season)))
ax2.set_yticks(np.arange(len(var_new)))
ax2.set_xticklabels(season2,rotation=0)
ax2.set_yticklabels(var_new)
# for edge, spine in ax2.spines.items():
#     spine.set_visible(False)
plt.setp(ax2.get_xticklabels(), rotation=0, ha="right",
         rotation_mode="anchor")

# ax2.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
# ax2.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
# ax2.grid(which="minor", color="w", linestyle='-', linewidth=3)
# ax2.tick_params(which="minor", bottom=False, left=False)

for i in range(len(var_new)):
    for j in range(len(season)):
        if round(total_night[:,:,1].T[i, j])<0.05:
            te=str(('%.2f'%total_night[:,:,0].T[i, j]))+'*'
            text = ax2.text(j, i, te,
                ha="center", va="center", color="k",fontsize=15 )
        else: 
            text= ax2.text(j, i, ('%.2f'%total_night[:,:,0].T[i, j]),
                           ha="center", va="center", color="k",fontsize=15)
ax2.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
ax2.tick_params(labelleft=False)
ax2.set_xticklabels(season2,fontsize=16)
ax2.set_xlabel('Nighttime',fontsize=20)
ax2.text(-0.1, 1.0, 'b', fontsize=20, transform=ax2.transAxes, fontweight='bold')

im3 = ax3.imshow(total_ndvi[:,0].T.reshape(11,1),vmin=-0.15,vmax=0.15,cmap='bwr',aspect=1)
ax3.set_yticks(np.arange(len(var_new)))
ax3.set_xticklabels([' ',r'$\Delta$NDVI',' '],rotation=0,fontsize=16)
ax3.set_yticklabels(var_new)
# for edge, spine in ax2.spines.items()
#     spine.set_visible(False)
plt.setp(ax3.get_xticklabels(), rotation=0, ha="right",
         rotation_mode="anchor")
ax3.tick_params(top=True, bottom=False,labeltop=True, labelbottom=False)
ax3.tick_params(labelleft=False)
ax3.text(-0.5, 1.0, 'c', fontsize=20, transform=ax3.transAxes, fontweight='bold')

for i in range(len(var_new)):
    if total_ndvi[:,1].T.reshape(11,1)[i]<0.05:
             te=str(('%.2f'%total_ndvi[:,0].T.reshape(11,1)[i]))+'*'
             text = ax3.text(0, i, te,ha="center", va="center", color="k",fontsize=15)
    else: 
             text= ax3.text(0, i, ('%.2f'%total_ndvi[:,0].T.reshape(11,1)[i]),
                            ha="center", va="center", color="k",fontsize=15)
            
cax = fig.add_axes([ax3.get_position().x1+0.03,ax3.get_position().y0,0.03,ax3.get_position().height])
cbar = plt.colorbar(im3, cax=cax) # Similar to fig.colorbar(im, cax = cax)
# cbar = ax3.figure.colorbar(im3, ax=ax3,cmap="bwr")
cbar.ax.set_yticklabels(['-0.15','-0.10','-0.05','0','0.05','0.10','0.15'],fontsize=16)
cbar.ax.set_ylabel('Correlation',fontsize=20)
# fig.tight_layout()#调整整体空白
# plt.subplots_adjust(wspace =0.2, hspace =0)#调整子图间距
plt.savefig('../results/figures/fig8.spatial correlations between lst effects and factors09012.jpg',bbox_inches='tight',dpi=300)
