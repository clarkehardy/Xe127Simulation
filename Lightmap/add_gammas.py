import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import histlite as hl
import pandas as pd
import LightMap
import sys
import os
import pickle
import argparse
import time
import gzip
from mpl_toolkits.axes_grid1 import make_axes_locatable

sim_dir = os.getenv('SIM_DIR')

parser = argparse.ArgumentParser()
parser.add_argument('-input_files',type=str,nargs='+')

args = parser.parse_args()
input_files = args.input_files

m = -0.1
b = 9600
def peak_sep(x):
    return m*x+b

# load TPC used for training
print('\nLoading TPC geometry used for lightmap training...\n')
with open(sim_dir+'tpc.pkl', 'rb') as handle:
    tpc = pickle.load(handle)
print(tpc)

# redefine TPC as reduced volume within field rings and between cathode and anode
tpc.r = 566.65
tpc.zmax = -402.97 #tpc.zmax-19.#1199#17#19.
tpc.zmin = -1585.97#tpc.zmax-1183.#3#21#1183.

# load model
print('\nLoading true lightmap model...\n')
lm_true = LightMap.load_model(sim_dir+'true-lm', 'LightMapHistRZ')
print(lm_true, '\n')

# function to plot scatter plot with histograms
def proj2d(x,y,xlabel='x',ylabel='y',bins=200,s=0.001,color='blue'):
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.02
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    f = plt.figure(figsize=(5, 5))
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)
    ax_scatter.scatter(x, y, color=color, s=s, alpha=0.1)
    xmin = 0
    ymin = 0
    xmax = 100000
    ymax = 100000
    binsx = np.linspace(xmin,xmax,bins)
    binsy = np.linspace(ymin,ymax,bins)
    ax_scatter.set_xlim((xmin,xmax))
    ax_scatter.set_ylim((ymin,ymax))
    ax_scatter.set_xlabel(xlabel)
    ax_scatter.set_ylabel(ylabel)
    ax_histx.hist(x, bins=binsx, color=color, histtype=u'step')
    ax_histy.hist(y, bins=binsy, color=color, orientation='horizontal', histtype=u'step')
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())
    ax_histx.set_yticks([])
    ax_histy.set_xticks([])
    return f,ax_histx,ax_histy,ax_scatter


# *********************************************************************************************************
# LOOP THROUGH ALL DATASETS
# *********************************************************************************************************

# loop through and collect the specified number of events
data = []
print('Collecting events from {:d} processed simulation files...\n'.format(len(input_files)))
for data_file in input_files:
    input_file = gzip.open(data_file,'rb')
    this_df = pickle.load(input_file)
    data.append(this_df)
    input_file.close()

# add to pandas dataframe
data = pd.concat(data,ignore_index=True)

print('Number of events which produce a charge signal: {}'.format(len(data.index)))

#print('Sampling {:d} events for calibration '.format(events)+val_string+'using seed {:d}...\n'.format(seed))
#data = data.sample(val_factor*events, random_state=seed)

# compute z from the drift time and TPC dimensions
# drift velocity from 2021 sensitivity paper
# TPC center + (TPC length)/2 - TPC top to anode
# 1022.6 + 1277/2 - 18.87
data['z'] = -402.97 - data.weighted_drift.values*1.709

mask = (data['weighted_radius'] < tpc.r - 20.) & (data['z'] > tpc.zmin + 20.) & (data['z'] < tpc.zmax - 20.)
mask = mask & (data['Observed Light']>peak_sep(data['evt_charge_including_noise'])) & (data['Observed Light']<4200)

print('Number of events in 1300keV peak which survive cuts: {}'.format(len(data[mask].index)))

# plot MC truth scatter plot
fig,ax_histx,ax_histy,ax_scatter = proj2d(data.fNTE.values[mask],data.fInitNOP.values[mask],xlabel='MC Truth Ionization (Number of Electrons)',\
                                      ylabel='MC Truth Scintillation (Number of Photons)',s=0.2)

# plot charge and light with selection cut
xvals = np.linspace(0,90000,10)
yvals = peak_sep(xvals)
#plt.rcParams.update({'font.size': 18})
fig,ax = plt.subplots(figsize=(8,6))
ht = ax.hist2d(data.evt_charge_including_noise.values[mask],data['Observed Light'][mask],200,norm=mpl.colors.LogNorm())
ax.set_xlabel('Detected Electrons')
ax.set_ylabel('Detected Photons')
#ax.set_xlim([8000,28000])
#ax.set_ylim([200,1700])
ax.plot(xvals,yvals,'--r',lw=4)
cbar = fig.colorbar(ht[3])
fig.tight_layout()

# plot spatial distribution of events in r and z
fig,ax = plt.subplots(figsize=(3.5,5))
hist,r_edges,z_edges = np.histogram2d(data.weighted_radius.values[mask],data.z.values[mask],bins=50,density=True)
r_bins = (r_edges[:-1]+r_edges[1:])/2.
z_bins = (z_edges[:-1]+z_edges[1:])/2.
im = ax.imshow(hist.T*np.sum(mask)/(2*np.pi*r_bins[np.newaxis,:]),extent=[r_edges.min(),r_edges.max(),z_edges.min(),z_edges.max()])
cbar = fig.colorbar(im)
cbar.formatter.set_powerlimits((0, 0))
cbar.set_label('Event Density (events/mm$^3$)')
ax.set_xlim([0,tpc.r])
ax.set_ylim([tpc.zmin,tpc.zmax])
ax.set_xlabel(r'$r$ (mm)')
ax.set_ylabel(r'$z$ (mm)')
ax.set_title('Event Distribution')
ax.set_aspect('equal')
fig.tight_layout()

# plot spatial distribution x and y
fig,ax = plt.subplots(figsize=(5,5))
hist,x_edges,y_edges = np.histogram2d(data.weighted_x.values[mask],data.weighted_y.values[mask],bins=200,density=True)
x_bins = (x_edges[:-1]+x_edges[1:])/2.
y_bins = (y_edges[:-1]+y_edges[1:])/2.
im = ax.imshow(hist.T*np.sum(mask)/(tpc.zmax-tpc.zmin),extent=[x_edges.min(),x_edges.max(),y_edges.min(),y_edges.max()])
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im,cax=cax)
cbar.formatter.set_powerlimits((0, 0))
cbar.set_label('Event Density (events/mm$^3$)')
ax.set_xlabel(r'$x$ (mm)')
ax.set_ylabel(r'$y$ (mm)')
ax.set_title('Event Distribution')
ax.set_aspect('equal')
fig.tight_layout()

plt.show()
