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
ndvi_peak=real['ndvi_peak_trend_8_10']*5
ndvi_peak_str='mean='+str(int(round((ndvi_peak*10000).mean(),0)))

# Load data
#shapefile='../data/shp/cb_2018_us_county_20m/cb_2018_us_county_20m.shp'
shapefile='../data/shp/cb_2018_us_state_20m/cb_2018_us_state_20m.shp'
county_shapes = shpreader.Reader(shapefile)
state_shapes = shpreader.Reader(shapefile)
po=format(len(ndvi_peak[ndvi_peak>0])/len(ndvi_peak.dropna()), '.2%')
ne=format(len(ndvi_peak[ndvi_peak<0])/len(ndvi_peak.dropna()), '.2%')
cmaps = colors.ListedColormap(['#87CEFA','#008000','#DC143C','#0000FF'])
mycolor = colors.LinearSegmentedColormap.from_list('mycmap', ['SaddleBrown', 'white', 'DarkGreen'])

sns.set_palette("hls") 
lons=real['xlong']
lats=real['ylat']
# make plot
projection = ccrs.Mercator()
axes_class = (GeoAxes,
              dict(map_projection=projection))

fig = plt.figure(figsize=(18, 12))

grid = AxesGrid(fig, 111, axes_class=axes_class,
                 nrows_ncols=(1, 1),
                 axes_pad=0.6,  # pad between axes in inch.

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

    

p = grid[0].scatter(lons, lats, 
               c=ndvi_peak*10000*10,
               s=88,
               vmin=-5000,
               vmax=5000,
                transform=ccrs.PlateCarree(),
                cmap=mycolor,
                zorder=10)
add_map_lines(grid[0])

#add kde plot
ax1 = grid[0].inset_axes([0.825,0.15,0.15,0.25],transform=grid[0].transAxes)

N, bins, patches = ax1.hist(ndvi_peak*10000*10/2, 40,edgecolor='white', linewidth=1)
for i in range(0,18):
    patches[i].set_facecolor('SaddleBrown')
for i in range(18, len(patches)):
    patches[i].set_facecolor('DarkGreen')

#ax1.vlines([0, 0], 0, 0.9, transform=ax1.get_xaxis_transform(), colors='r',linestyles = "dashed")
ax1.annotate(po, xy=(2500, 5), 
             xytext=(2500, 25),
             fontsize=20,
             c='DarkGreen')
ax1.annotate(ne, xy=(-2500, 5), 
             xytext=(-13000, 25),
             fontsize=20,
             c='SaddleBrown')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.yaxis.set_visible(False)
ax1.set_xlim(-10000,10000)
ax1.set_xticks(np.arange(-10000,10001,10000))
ax1.set_xticklabels(np.arange(-10000,10001,10000),fontsize=14)
#ax1.set_xlabel('Distribution',fontsize=20)
ax1.text(0.12,1,ndvi_peak_str,fontsize=18,transform=ax1.transAxes,c='SaddleBrown')
ax1.set_facecolor('none')
ax1.set_xlabel(r'$\Delta$NDVI',fontsize=24)

cb = grid.cbar_axes[0].colorbar(p,ticks=np.arange(-5000, 5500 ,1000))
cb.ax.tick_params(axis='y',direction='in')
cb.ax.set_yticklabels(np.arange(-5000, 5500 ,1000),fontsize=14)
cb.set_label_text(r'$\Delta$NDVI',fontsize=30)


plt.savefig('../results/figures/figure/1202/fig4.ndvi effects10.14.jpg',bbox_inches='tight',dpi=300)

