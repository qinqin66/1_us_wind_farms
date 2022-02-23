#!/usr/bin/env python
# coding: utf-8
#import libs
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

#import data and file
data=pd.read_csv('../data/data_whole_add_ndvipeak.csv')
real=data[data['wf_end_builit_year0.95']<2018][data['wf_end_builit_year0.95']>0]
shapefile='../data/shp/cb_2018_us_state_20m/cb_2018_us_state_20m.shp'
state_shapes = shpreader.Reader(shapefile)

#calculate the effets in 5 years time window
day_annual=real.iloc[:,156:168].mean(1)*5
night_annual=real.iloc[:,168:180].mean(1)*5

#text
day_annual_str='mean='+str(round(day_annual.mean(),2))
night_annual_str='mean='+str('%.2f'%round(night_annual.mean(),2))
day_po=format(len(day_annual[day_annual>0])/len(day_annual.dropna()),'.2%')
day_ne=format(len(day_annual[day_annual<0])/len(day_annual.dropna()),'.2%')
night_po=format(len(night_annual[night_annual>0])/len(day_annual.dropna()),'.2%')
night_ne=format(len(night_annual[night_annual<0])/len(day_annual.dropna()),'.2%')

#plot setting
sns.set_palette("hls")
lons=real['xlong']
lats=real['ylat']
# make plot
projection = ccrs.Mercator()
axes_class = (GeoAxes,
              dict(map_projection=projection))

fig = plt.figure(figsize=(18, 24))

grid = AxesGrid(fig, 111, axes_class=axes_class,
                 nrows_ncols=(2, 1),
                 axes_pad=1.0,  # pad between axes in inch.

                 cbar_location='right',
                 cbar_mode='single',
                 cbar_pad=0.2,
                 cbar_size='2%',
                 share_all=True,
                 label_mode=" ")


def add_map_lines(ax):
    for state in state_shapes.geometries():
        ax.add_geometries([state], ccrs.PlateCarree(),
                          facecolor='white', edgecolor='k',linewidth=1.5) 
    ax.coastlines(resolution='50m', lw=1.5)
    ax.set_extent([-128, -65, 24, 45], ccrs.Geodetic())
    ax.set_xticks(np.linspace(-125,-65, 3), crs=ccrs.PlateCarree())
    ax.set_xticklabels(np.linspace(-125,-65, 3),fontsize=16)
    ax.set_yticks(np.linspace(25, 45, 3), crs=ccrs.PlateCarree())
    ax.set_yticklabels(np.linspace(25, 45, 3),fontsize=16)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

#day
p = grid[0].scatter(lons, lats, 
               c=day_annual,
               s=88,
               vmin=-0.5,
               vmax=0.5,
                transform=ccrs.PlateCarree(),
                cmap='bwr',
                zorder=10)
add_map_lines(grid[0])
grid[0].xaxis.set_visible(False)
grid[0].text(-0.05, 1.02, 'a', fontsize=36, transform=grid[0].transAxes, fontweight='bold')
grid[0].set_title(r"Daytime",fontsize=36)

#add kde plot
ax1 = grid[0].inset_axes([0.825,0.15,0.15,0.25],transform=grid[0].transAxes)
N, bins, patches = ax1.hist(day_annual, 21,edgecolor='white', linewidth=1)
for i in range(0,8):
    patches[i].set_facecolor('b')
for i in range(8, len(patches)):
    patches[i].set_facecolor('r')

#ax1.vlines([0, 0], 0, 0.9, transform=ax1.get_xaxis_transform(), colors='k',linestyles = "dashed")
ax1.annotate(day_po, xy=(0.75, 48), 
             fontsize=16,
             c='r')
ax1.annotate(day_ne, xy=(-2.5, 48), 
             fontsize=16,
             c='b')
ax1.text(0.12,1,day_annual_str,fontsize=18,transform=ax1.transAxes,c='r')

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.yaxis.set_visible(False)
ax1.set_xlim(-2.5,2.5)
ax1.set_xticks([-2.5,0,2.5])
ax1.set_xticklabels([-2.5,0,2.5],fontsize=14)
ax1.set_facecolor('none')
ax1.set_xlabel(r'$\Delta$LST(℃)',fontsize=24)

#night
p2 = grid[1].scatter(lons, lats, 
               c=night_annual,
               s=88,
#               marker='s',
               vmin=-0.5,
               vmax=0.5,
                transform=ccrs.PlateCarree(),
                cmap='bwr',
                zorder=10)
add_map_lines(grid[1])
grid[1].text(-0.05, 1.02, 'b', fontsize=36, transform=grid[1].transAxes, fontweight='bold')
grid[1].set_title(r"Nighttime",fontsize=36)
#add kde plot
ax2 = grid[1].inset_axes([0.825,0.15,0.15,0.25],transform=grid[1].transAxes)

N, bins, patches = ax2.hist(night_annual, 14,edgecolor='white', linewidth=1)
for i in range(0,6):
    patches[i].set_facecolor('b')
for i in range(6, len(patches)):
    patches[i].set_facecolor('r')

#ax2.vlines([0, 0], 0, 0.9, transform=ax2.get_xaxis_transform(), colors='k',linestyles = "dashed")
ax2.annotate(night_po, xy=(0.75, 48), 
             fontsize=16,
             c='r')
ax2.annotate(night_ne, xy=(-2.5, 48), 
             fontsize=16,
             c='b')
ax2.text(0.12,1,night_annual_str,fontsize=18,transform=ax2.transAxes,c='r')

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.yaxis.set_visible(False)
ax2.set_xlim(-2.5,2.5)
ax2.set_xticks([-2.5,0,2.5])
ax2.set_xticklabels([-2.5,0,2.5],fontsize=14)
ax2.set_facecolor('none')
ax2.set_xlabel(r'$\Delta$LST($^\circ$C)',fontsize=24)


cb = grid.cbar_axes[0].colorbar(p,ticks=[-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0])
cb.ax.tick_params(axis='y',direction='in')
cb.ax.set_yticklabels([-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0],fontsize=18)
cb.set_label_text(r'$\Delta$LST($^\circ$C)',fontsize=36)

# save figure
plt.savefig('../results/figures/fig1.annual lst effect.jpg',bbox_inches='tight',dpi=300)
