#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.io.shapereader as shpreader
import matplotlib.colors as colors

data=pd.read_csv('../data/data_whole_add_ndvipeak.csv')
real=data[data['wf_end_builit_year0.95']<2018][data['wf_end_builit_year0.95']>0]
shapefile='../data/shp/cb_2018_us_state_20m/cb_2018_us_state_20m.shp'
state_shapes = shpreader.Reader(shapefile)

night_mam=(real.iloc[:,170:173].mean(1)*5).dropna()
real = real.loc[night_mam.dropna().index]
night_jja=real.iloc[:,173:176].mean(1)*5
night_djf=pd.concat([real.iloc[:,179:180],real.iloc[:,168:170]],axis=1).mean(1)*5
night_son=real.iloc[:,176:179].mean(1)*5
night_mam_str='mean='+str(round(night_mam.mean(),2))
night_jja_str='mean='+str(round(night_jja.mean(),2))
night_son_str='mean='+str(round(night_son.mean(),2))
night_djf_str='mean='+str(round(night_djf.mean(),2))

mam_po=format(len(night_mam[night_mam>0])/len(night_mam.dropna()),'.2%')
mam_ne=format(len(night_mam[night_mam<0])/len(night_mam.dropna()),'.2%')
jja_po=format(len(night_jja[night_jja>0])/len(night_jja.dropna()),'.2%')
jja_ne=format(len(night_jja[night_jja<0])/len(night_jja.dropna()),'.2%')
son_po=format(len(night_son[night_son>0])/len(night_son.dropna()),'.2%')
son_ne=format(len(night_son[night_son<0])/len(night_son.dropna()),'.2%')
djf_po=format(len(night_djf[night_djf>0])/len(night_djf.dropna()),'.2%')
djf_ne=format(len(night_djf[night_djf<0])/len(night_djf.dropna()),'.2%')

var=['lst_trend_night_jan_buff8_10','lst_trend_night_feb_buff8_10','lst_trend_night_mar_buff8_10',
    'lst_trend_night_apr_buff8_10','lst_trend_night_may_buff8_10','lst_trend_night_jun_buff8_10',
    'lst_trend_night_jul_buff8_10','lst_trend_night_aug_buff8_10','lst_trend_night_sep_buff8_10',
    'lst_trend_night_oct_buff8_10','lst_trend_night_nov_buff8_10','lst_trend_night_dec_buff8_10']
night=real[var]
ni=np.array(np.abs(night))
night['max_value']=night.max(axis=1)
night['max_index']=np.argmax(ni,axis=1)+1
night['min_value']=night.min(axis=1)
night['min_index']=np.argmin(ni,axis=1)+1
gath= pd.concat([pd.DataFrame(night_mam,columns=['MAM']),
           pd.DataFrame(night_jja,columns=['JJA']),
           pd.DataFrame(night_son,columns=['SON']),
           pd.DataFrame(night_djf,columns=['DJF'])],axis=1)
ga=np.array(np.abs(gath))
gath['max_value']=gath.max(axis=1)
gath['max_index']=np.argmax(ga,axis=1)+1
gath['min_value']=gath.min(axis=1)
gath['min_index']=np.argmin(ga,axis=1)+1
night_min_mam=str(format(len(gath[gath['min_index']==1])/len(gath.dropna()),'.2%'))
night_min_jja=str(format(len(gath[gath['min_index']==2])/len(gath.dropna()),'.2%'))
night_min_son=str(format(len(gath[gath['min_index']==3])/len(gath.dropna()),'.2%'))
night_min_djf=str(format(len(gath[gath['min_index']==4])/len(gath.dropna()),'.2%'))
night_max_mam=str(format(len(gath[gath['max_index']==1])/len(gath.dropna()),'.2%'))
night_max_jja=str(format(len(gath[gath['max_index']==2])/len(gath.dropna()),'.2%'))
night_max_son=str(format(len(gath[gath['max_index']==3])/len(gath.dropna()),'.2%'))
night_max_djf=str(format(len(gath[gath['max_index']==4])/len(gath.dropna()),'.2%'))
cmaps = colors.ListedColormap(['#87CEFA','#008000','#DC143C','#0000FF'])

sns.set_palette("hls")
lons=real['xlong']
lats=real['ylat']
# make plot
projection = ccrs.Mercator()
axes_class = (GeoAxes,
              dict(map_projection=projection))
fig = plt.figure(figsize=(18, 24))
grid = AxesGrid(fig, 111, axes_class=axes_class,
                 nrows_ncols=(3, 2),
                 axes_pad=(0.3, 0.6),  # pad between axes in inch.
                 cbar_location='right',
                 cbar_mode='each',
                 cbar_pad=0.2,
                 cbar_size='2%',
                 share_all=True,
                 label_mode=" ")

def add_map_lines(ax):
    for state in state_shapes.geometries():
        ax.add_geometries([state], ccrs.PlateCarree(),
                          facecolor='white', edgecolor='k',linewidth=1.0) 
    ax.coastlines(resolution='50m', lw=1.0)
    ax.set_extent([-128, -65, 24, 45], ccrs.Geodetic())
    ax.set_xticks(np.linspace(-125,-65, 3), crs=ccrs.PlateCarree())
    ax.set_xticklabels(np.linspace(-125,-65, 3),fontsize=16)
    ax.set_yticks(np.linspace(25, 45, 3), crs=ccrs.PlateCarree())
    ax.set_yticklabels(np.linspace(25, 45, 3),fontsize=16)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    
#mam
p = grid[0].scatter(lons, lats, 
               c=night_mam,
               s=30,
               vmin=-0.5,
               vmax=0.5,
                transform=ccrs.PlateCarree(),
                cmap='bwr',
                zorder=10)
add_map_lines(grid[0])
grid[0].xaxis.set_visible(False)
grid[0].text(-0.05, 1.02, 'a', fontsize=24, transform=grid[0].transAxes, fontweight='bold')
grid[0].set_title(r"Spring",fontsize=24)

#add kde plot
ax1 = grid[0].inset_axes([0.815,0.15,0.15,0.25],transform=grid[0].transAxes)
N, bins, patches = ax1.hist(night_mam, 21,edgecolor='white', linewidth=1)
for i in range(0,12):
    patches[i].set_facecolor('b')
for i in range(12, len(patches)):
    patches[i].set_facecolor('r')
#ax1.vlines([0, 0], 0, 0.9, transform=ax1.get_xaxis_transform(), colors='k',linestyles = "dashed")
ax1.text(0.65,0.6,mam_po,fontsize=10,c='r',transform=ax1.transAxes)
ax1.text(-0.15,0.6,mam_ne,fontsize=10,c='b',transform=ax1.transAxes)
ax1.text(0.15,1,night_mam_str,fontsize=12,transform=ax1.transAxes,c='b')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.yaxis.set_visible(False)
ax1.set_xlim(-2.5,2.5)
ax1.set_xticks([-2.5,0,2.5])
ax1.set_xticklabels([-2.5,0,2.5],fontsize=10)
ax1.set_facecolor('none')
ax1.set_xlabel(r'$\Delta$LST(℃)',fontsize=12)

#jja
p2 = grid[1].scatter(lons, lats, 
               c=night_jja,
               s=30,
               vmin=-0.5,
               vmax=0.5,
                transform=ccrs.PlateCarree(),
                cmap='bwr',
                zorder=10)
add_map_lines(grid[1])
grid[1].xaxis.set_visible(False)
grid[1].yaxis.set_visible(False)
grid[1].text(-0.05, 1.02, 'b', fontsize=24, transform=grid[1].transAxes, fontweight='bold')
grid[1].set_title(r"Summer",fontsize=24)
#add kde plot
ax2 = grid[1].inset_axes([0.815,0.15,0.15,0.25],transform=grid[1].transAxes)
N, bins, patches = ax2.hist(night_jja, 21,edgecolor='white', linewidth=1)
for i in range(0,16):
    patches[i].set_facecolor('b')
for i in range(16, len(patches)):
    patches[i].set_facecolor('r')
#ax2.vlines([0, 0], 0, 0.9, transform=ax2.get_xaxis_transform(), colors='k',linestyles = "dashed")
ax2.text(0.66,0.6,jja_po,fontsize=10,c='r',transform=ax2.transAxes)
ax2.text(-0.1,0.6,jja_ne,fontsize=10,c='b',transform=ax2.transAxes)
ax2.text(0.15,1,night_jja_str,fontsize=12,transform=ax2.transAxes,c='r')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.yaxis.set_visible(False)
ax2.set_xlim(-2.5,2.5)
ax2.set_xticks([-2.5,0,2.5])
ax2.set_xticklabels([-2.5,0,2.5],fontsize=10)
ax2.set_facecolor('none')
ax2.set_xlabel(r'$\Delta$LST(℃)',fontsize=12)

#son
p3 = grid[2].scatter(lons, lats, 
               c=night_son,
               s=30,
               vmin=-0.5,
               vmax=0.5,
                transform=ccrs.PlateCarree(),
                cmap='bwr',
                zorder=10)
add_map_lines(grid[2])
grid[2].xaxis.set_visible(False)
grid[2].text(-0.05, 1.02, 'c', fontsize=24, transform=grid[2].transAxes, fontweight='bold')
grid[2].set_title(r"Autumn",fontsize=24)

#add kde plot
ax3 = grid[2].inset_axes([0.815,0.15,0.15,0.25],transform=grid[2].transAxes)
N, bins, patches = ax3.hist(night_son, 21,edgecolor='white', linewidth=1)
for i in range(0,9):
    patches[i].set_facecolor('b')
for i in range(9, len(patches)):
    patches[i].set_facecolor('r')

#ax3.vlines([0, 0], 0, 0.9, transform=ax3.get_xaxis_transform(), colors='k',linestyles = "dashed")
ax3.text(0.65,0.6,son_po,fontsize=10,c='r',transform=ax3.transAxes)
ax3.text(-0.15,0.6,son_ne,fontsize=10,c='b',transform=ax3.transAxes)
ax3.text(0.15,1,night_son_str,fontsize=12,transform=ax3.transAxes,c='r')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.yaxis.set_visible(False)
ax3.set_xlim(-2.5,2.5)
ax3.set_xticks([-2.5,0,2.5])
ax3.set_xticklabels([-2.5,0,2.5],fontsize=10)
ax3.set_facecolor('none')
ax3.set_xlabel(r'$\Delta$LST(℃)',fontsize=12)

#djf
p4 = grid[3].scatter(lons, lats, 
               c=night_djf,
               s=30,
               vmin=-0.5,
               vmax=0.5,
                transform=ccrs.PlateCarree(),
                cmap='bwr',
                zorder=10)
add_map_lines(grid[3])
grid[3].xaxis.set_visible(False)
grid[3].yaxis.set_visible(False)
grid[3].text(-0.05, 1.02, 'd', fontsize=24, transform=grid[3].transAxes, fontweight='bold')
grid[3].set_title(r"Winter",fontsize=24)
#add kde plot
ax4 = grid[3].inset_axes([0.815,0.15,0.15,0.25],transform=grid[3].transAxes)

N, bins, patches = ax4.hist(night_djf, 31,edgecolor='white', linewidth=1)
for i in range(0,12):
    patches[i].set_facecolor('b')
for i in range(12, len(patches)):
    patches[i].set_facecolor('r')
#ax4.vlines([0, 0], 0, 0.9, transform=ax4.get_xaxis_transform(), colors='k',linestyles = "dashed")
ax4.text(0.65,0.6,djf_po,fontsize=10,c='r',transform=ax4.transAxes)
ax4.text(-0.26,0.6,djf_ne,fontsize=10,c='b',transform=ax4.transAxes)
ax4.text(0.15,1,night_djf_str,fontsize=12,transform=ax4.transAxes,c='r')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_visible(False)
ax4.yaxis.set_visible(False)
ax4.set_xlim(-2.5,2.5)
ax4.set_xticks([-2.5,0,2.5])
ax4.set_xticklabels([-2.5,0,2.5],fontsize=10)
ax4.set_facecolor('none')
ax4.set_xlabel(r'$\Delta$LST(℃)',fontsize=12)


p5 = grid[4].scatter(lons, lats, 
               c=gath['min_index'],
               s=30,
               vmin=1,
               vmax=4,
                transform=ccrs.PlateCarree(),
                cmap=cmaps,
                zorder=10)
add_map_lines(grid[4])
grid[4].set_title(r"Minimum LST effect",fontsize=24)
grid[4].text(-0.05, 1.02, 'e', fontsize=24, transform=grid[4].transAxes, fontweight='bold')
#grid[4].text(-0.2, 0.3,r"Nighttime",fontsize=24,transform=grid[4].transAxes,rotation='vertical')
#grid[2].set_title(r"SON",fontsize=24)

#add kde plot
ax5 = grid[4].inset_axes([0.815,0.08,0.15,0.25],transform=grid[4].transAxes)
N, bins, patches = ax5.hist(night['min_index'], 12,edgecolor='white', linewidth=1)
for i in range(2,5):
     patches[i].set_facecolor('#87CEFA')
for i in range(5,8):
     patches[i].set_facecolor('#008000')
for i in range(8,11):
     patches[i].set_facecolor('#DC143C')
for i in range(11, len(patches)):
     patches[i].set_facecolor('#0000FF')
for i in range(0, 2):
     patches[i].set_facecolor('#0000FF')

#ax3.vlines([0, 0], 0, 0.9, transform=ax3.get_xaxis_transform(), colors='k',linestyles = "dashed")
# ax3.text(0.65,0.6,son_po,fontsize=10,c='r',transform=ax3.transAxes)
# ax3.text(-0.26,0.6,son_ne,fontsize=10,c='b',transform=ax3.transAxes)
# ax3.text(0.12,1,night_son_str,fontsize=12,transform=ax3.transAxes)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['left'].set_visible(False)
ax5.yaxis.set_visible(False)
ax5.xaxis.set_tick_params(direction='in')
ax5.set_xticklabels('')
ax5.set_xlabel(r'Month',fontsize=10)

# ax3.set_xlim(-3,3)
# ax3.set_xticks([-3.0,0,3.0])
# ax3.set_xticklabels([-3.0,0,3.0],fontsize=10)
ax5.set_facecolor('none')
#ax3.set_xlabel(r'$\Delta$LST(℃)',fontsize=12)
ax5.text(-0.08, 0.75, night_min_mam, fontsize=10, transform=ax5.transAxes,c='#00BFFF')
ax5.text(0.4, 1, night_min_jja, fontsize=10, transform=ax5.transAxes,c='#008000')
ax5.text(0.65, 0.65, night_min_son, fontsize=10, transform=ax5.transAxes,c='#DC143C')
ax5.text(-0.4, 0.45, night_min_djf, fontsize=10, transform=ax5.transAxes,c='#0000FF')

#night max
p6 = grid[5].scatter(lons, lats, 
               c=gath['max_index'],
               s=30,
               vmin=1,
               vmax=4,
                transform=ccrs.PlateCarree(),
                cmap=cmaps,
                zorder=10)
add_map_lines(grid[5])
grid[5].yaxis.set_visible(False)
grid[5].set_title(r"Maximum LST effect",fontsize=24)
grid[5].text(-0.05, 1.02, 'f', fontsize=24, transform=grid[5].transAxes, fontweight='bold')
#grid[3].set_title(r"DJF",fontsize=24)
#add kde plot
ax6 = grid[5].inset_axes([0.815,0.08,0.15,0.25],transform=grid[5].transAxes)

N, bins, patches = ax6.hist(night['max_index'], 12,edgecolor='white', linewidth=1)
for i in range(2,5):
     patches[i].set_facecolor('#87CEFA')
for i in range(5,8):
     patches[i].set_facecolor('#008000')
for i in range(8,11):
     patches[i].set_facecolor('#DC143C')
for i in range(11, len(patches)):
     patches[i].set_facecolor('#0000FF')
for i in range(0, 2):
     patches[i].set_facecolor('#0000FF')
#ax4.vlines([0, 0], 0, 0.9, transform=ax4.get_xaxis_transform(), colors='k',linestyles = "dashed")
# ax4.text(0.65,0.6,djf_po,fontsize=10,c='r',transform=ax4.transAxes)
# ax4.text(-0.26,0.6,djf_ne,fontsize=10,c='b',transform=ax4.transAxes)
# ax4.text(0.12,1,night_djf_str,fontsize=12,transform=ax4.transAxes)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['left'].set_visible(False)
ax6.yaxis.set_visible(False)
#ax4.xaxis.set_visible(False)
ax6.xaxis.set_tick_params(direction='in')
ax6.set_xticklabels('')
ax6.set_xlabel(r'Month',fontsize=10)
# ax4.set_xlim(-5,5)
# ax4.set_xticks(np.arange(1.5,12,0.75))
# ax4.set_xticklabels(np.arange(1,13,1),fontsize=8)
# ax4.set_facecolor('none')
# ax4.set_xlabel(r'$\Delta$LST(℃)',fontsize=12)
ax6.text(0.16, 0.8, night_max_mam, fontsize=10, transform=ax6.transAxes,c='#00BFFF')
ax6.text(0.28, 0.5, night_max_jja, fontsize=10, transform=ax6.transAxes,c='#008000')
ax6.text(0.6, 1, night_max_son, fontsize=10, transform=ax6.transAxes,c='#DC143C')
ax6.text(-0.47, 0.4, night_max_djf, fontsize=10, transform=ax6.transAxes,c='#0000FF')


grid.cbar_axes[0].set_visible(False)
grid.cbar_axes[2].set_visible(False)
grid.cbar_axes[4].set_visible(False)

cb = grid.cbar_axes[1].colorbar(p2,ticks=[-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
cb.ax.tick_params(axis='y',direction='in')
cb.ax.set_yticklabels([-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0],fontsize=14)
cb.set_label_text(r'$\Delta$LST(℃)',fontsize=30)

cb1 = grid.cbar_axes[3].colorbar(p4,ticks=[-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
cb1.ax.tick_params(axis='y',direction='in')
cb1.ax.set_yticklabels([-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0],fontsize=14)
cb1.set_label_text(r'$\Delta$LST(℃)',fontsize=30)

cb2 = grid.cbar_axes[5].colorbar(p6,ticks=[1.5,2.25,3,3.75])
cb2.ax.set_yticklabels(['Spring','Summer','Autumn','Winter'],fontsize=12,rotation='vertical')
cb2.ax.tick_params(axis='y',direction='in')
cb2.set_label_text(r'Seasons',fontsize=30)

plt.title('')
plt.savefig('../results/figures/fig3.nighttime lst effect0511.jpg',bbox_inches='tight',dpi=300)

