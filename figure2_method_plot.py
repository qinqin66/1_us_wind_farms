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


data=pd.read_csv('../data/data_whole_add_ndvipeak.csv')
real=data[data['wf_end_builit_year0.95']<2018][data['wf_end_builit_year0.95']>0]

shapefile='../data/shp/cb_2018_us_county_20m/cb_2018_us_county_20m.shp'
county_shapes = shpreader.Reader(shapefile)
shapefile='../data/shp/cb_2018_us_state_20m/cb_2018_us_state_20m.shp'
state_shapes = shpreader.Reader(shapefile)

#divide wind farm size
real['p_tnum'][real['p_tnum']<=25]=1
real['p_tnum'][real['p_tnum']>75]=3
real['p_tnum'][real['p_tnum']>3]=2

real = real[['xlong','ylat','land_cover','p_tnum']]
cover_forest=real[real['land_cover']==4.0]                      #38
cover_grass=real[real['land_cover']==10.0]                       #127
cover_crop=real[real['land_cover']==12.0]                      #115
cover_other=real[real['land_cover']!=12.0][real['land_cover']!=10.0][real['land_cover']!=4.0]  #40

real['land_cover'][real['land_cover']==4.0] = str('Forest')
real['land_cover'][real['land_cover']==10.0] = str('Grassland')                       #127
real['land_cover'][real['land_cover']==12.0] =  str('Cropland')                   #115
real['land_cover'][real['land_cover']==1] = str('Others')
real['land_cover'][real['land_cover']==2] = str('Others')
real['land_cover'][real['land_cover']==3] = str('Others')
real['land_cover'][real['land_cover']==5] = str('Others')
real['land_cover'][real['land_cover']==6] = str('Others')
real['land_cover'][real['land_cover']==7] = str('Others')
real['land_cover'][real['land_cover']==8] = str('Others')
real['land_cover'][real['land_cover']==9] = str('Others')
real['land_cover'][real['land_cover']==11] = str('Others')
real['land_cover'][real['land_cover']==13] = str('Others')
real['p_tnum'] = real['p_tnum']

sns.set_palette("hls") 
# make plot
projection = ccrs.Mercator()
axes_class = (GeoAxes,
              dict(map_projection=projection))

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

    
def add_cartopy(data1,data2,data3,data4):   
    fig = plt.figure(figsize=(18, 18))
    grid = AxesGrid(fig, 111, axes_class=axes_class,
                     nrows_ncols=(1, 1),
                     axes_pad=0.6,  # pad between axes in inch.
#                     cbar_location='right',
#                     cbar_mode='single',
 #                    cbar_pad=0.2,
  #                   cbar_size='2%',
                     share_all=True,
                     label_mode=" ")

    p = grid[0].scatter(data1['xlong'], data1['ylat'], 
                   c='#FFA500',
                    marker='v',
                   s=data1['p_tnum']*30,
                    transform=ccrs.PlateCarree(),
                    label='Grassland(127)',
#                    cmap='bwr',
                    zorder=10) 
    p1 = grid[0].scatter(data2['xlong'], data2['ylat'], 
                    c='#228B22',
                    marker='o',
                    s=data2['p_tnum']*30,
                    transform=ccrs.PlateCarree(),
                    label='Forest(38)',
#                    cmap='bwr',
                    zorder=10) 
    p2 = grid[0].scatter(data3['xlong'], data3['ylat'], 
                    c='#FF0000',
                    marker='s',
                    s=data3['p_tnum']*30,
                    transform=ccrs.PlateCarree(),
                    label='Cropland(115)',
#                    cmap='bwr',
                    zorder=10) 
    p3 = grid[0].scatter(data4['xlong'], data4['ylat'], 
                    c='#1E90FF',
                    marker='D',
                    s=data4['p_tnum']*30,
                    transform=ccrs.PlateCarree(),
                    label='Others(39)',
#                    cmap='bwr',
                    zorder=10) 

    add_map_lines(grid[0])
 #   grid2 = grid[0].twinx()
#    grid2[0].set_extent([-128, -65, 24, 45], ccrs.Geodetic())
    grid[0].legend(loc='lower left',framealpha=0,fontsize=16)
#    grid[0].text(loc='lower right','large:100',framealpha=0,fontsize=16)
#    leg = grid[0].legend(loc='lower left', title="")
#    grid[0].legend(loc='lower right',framealpha=0,fontsize=22)
    grid[0].text(0.82, 0.12, 'Small:     108\nMedium: 106\nLarge:     105', fontsize=18, transform=grid[0].transAxes)
#    grid[0].set_title(r"Correlations",fontsize=20)
    
add_cartopy(cover_grass,cover_forest,cover_crop,cover_other)

#add_cartopy(cover_forest['land_cover'])
add_cartopy(cover_grass,cover_forest,cover_crop,cover_other)
plt.savefig('../results/figures/fig s1.land cover distribution10.14.jpg',bbox_inches='tight',dpi=300)

