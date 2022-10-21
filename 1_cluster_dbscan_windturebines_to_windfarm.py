'''
Code using DBSCAN tocluster 60,000+ wind turbines to 419 wind farms.
'''
#!/usr/bin/env python
# coding: utf-8

from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

shp = gpd.read_file('../data/1_us_windturbines/uswtdb_v2_3_20200109.shp')
io=pd.read_csv('../data/1_us_windturbines/uswtdb_v2_3_20200109.csv')
clusteringus = DBSCAN(eps=0.1, min_samples=5).fit(io.loc[:,['xlong','ylat']])
io['cluster']=clusteringus.labels_
shp['cluster'] = clusteringus.labels_
fig, ax=plt.subplots(1,figsize=(15,10))
ax.scatter(x=io['xlong'],y=io['ylat'],c=clusteringus.labels_,cmap='hsv')
ax.set_xlim([-120,-50])
ax.set_ylim([20,50])
shp.to_file('../data/1_us_windturbines/io_cluster.shp')

