import LightMap
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid
import pickle
import histlite as hl
import os
import pandas as pd
import sys
import cycler
from matplotlib.colors import LogNorm
from scipy.ndimage.filters import gaussian_filter

# ***********************************************************************************************************
# SET BASIC RUN OPTIONS HERE
# ***********************************************************************************************************

# read new data or retrieve saved data
read_again = False

# choose how to name plots
plot_name = 'for_Brian'

# choose whether to show error plot or efficiency plot
error_plot = True

# save figures here
home_dir = os.getenv('WORKING_DIR')
data_dir = os.getenv('DATA_DIR')
path = '{}/outputs/'.format(data_dir)

# set plotting style
plt.rc('figure', dpi=200, figsize=(4,3), facecolor='w')
plt.rc('savefig', dpi=200, facecolor='w')
plt.rc('lines', linewidth=1.5)

# ***********************************************************************************************************
# LOAD SAVED LIGHTMAPS FOR ANALYSIS
# ***********************************************************************************************************

# load true lightmap model
model_dir = '{}/lm-analysis'.format(home_dir)
lm_nn = LightMap.load_model(model_dir, 'LightMapNN')
print('\n', lm_nn, '\n')

# load TPC used for training
with open('tpc.pkl', 'rb') as handle:
    tpc = pickle.load(handle)
rang = (0, tpc.r), (tpc.zmin, tpc.zmax)
bins = 200
print(tpc)

# get histogram from true Lightmap
f_true = lambda r, z: lm_nn(r, np.repeat(0,np.shape(r)), z, cyl=True)
h_true = (hl.hist_from_eval(f_true, vectorize=False, bins=bins, range=rang)).values

if read_again==True:
    # define dataset sizes to be plotted
    sizes = [1e3,1e4,1e5,1e6,\
             1e3,1e4,1e5,1e6,\
             1e3,1e4,1e5,1e6,\
             1e3,1e4,1e5,1e6,\
             1e3,1e4,1e5,1e6,\
             1e3,1e4,1e5,1e6,\
             1e3,1e4,1e5,1e6,\
             1e3,1e4,1e5,1e6,\
             1e3,1e4,1e5,1e6]
    names = ['1k_final_1p_0','10k_final_1p_0','100k_final_1p_0','1M_final_1p_0',\
             '1k_final_1p_20','10k_final_1p_20','100k_final_1p_20','1M_final_1p_20',\
             '1k_final_1p_40','10k_final_1p_40','100k_final_1p_40','1M_final_1p_40',\
             '1k_final_2p_0','10k_final_2p_0','100k_final_2p_0','1M_final_2p_0',\
             '1k_final_2p_20','10k_final_2p_20','100k_final_2p_20','1M_final_2p_20',\
             '1k_final_2p_40','10k_final_2p_40','100k_final_2p_40','1M_final_2p_40',\
             'a1k_final_0','a10k_final_0','a100k_final_0','a1M_final_0',\
             'a1k_final_20','a10k_final_20','a100k_final_20','a1M_final_20',\
             'a1k_final_40','a10k_final_40','a100k_final_40','a1M_final_40']
    cut_dists = [0,0,0,0,20,20,20,20,40,40,40,40,\
                 0,0,0,0,20,20,20,20,40,40,40,40,\
                 0,0,0,0,20,20,20,20,40,40,40,40]
    typ = 'LightMapNN'

    data = pd.DataFrame()
    data['name'] = names
    data['size'] = sizes
    data['av_hist'] = [np.ndarray((bins,bins)) for _ in range(len(sizes))]
    data['bins'] = [np.ndarray((bins,bins)) for _ in range(len(sizes))]
    data['each_hist'] = [[] for _ in range(len(sizes))]
    data['cut_dist'] = np.array(cut_dists,dtype=float)
    
    # populate dataframe with histograms for each dataset
    for i in range(len(names)):
        lm = LightMap.load_model(path+'LightMap_'+names[i],typ)
        f = lambda r, z: lm(r, np.repeat(0,np.shape(r)), z, cyl=True)
        h = hl.hist_from_eval(f, vectorize=False, bins=bins, range=rang)
        data['av_hist'][i] = h.values
        data['bins'][i] = h._bins
        del f,h
        for j in range(len(lm.models)):
            lm_sub = LightMap.LightMapNN(tpc)
            lm_sub.models = [lm.models[j]]
            f = lambda r, z: lm_sub(r, np.repeat(0,np.shape(r)), z, cyl=True)
            h = hl.hist_from_eval(f, vectorize=False, bins=bins, range=rang)
            data['each_hist'][i].append(h.values)
            del lm_sub,f,h
        del lm
    
    # save data to be retrieved later
    data.to_pickle(path+'analysis.pkl')
else:
    data = pd.read_pickle(path+'analysis.pkl')

# ***********************************************************************************************************
# PRODUCE SOME PLOTS USING THE DATA
# ***********************************************************************************************************

# plot RMS error vs dataset size for various options
colls = ['final_1p_0','final_0']
descrips = ['Xe-127','Rn-222']
series_list = []
[series_list.append(data[[(series in name) for name in data['name']]]) for series in colls]
plt.figure(figsize=(5,4))
color = plt.cm.plasma(np.linspace(0, 1,len(colls)+1))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
plt.xlabel('Number of Events')
plt.ylabel('RMS Percent Error')
plt.axhline(1.,color='grey',ls='--')
for i in range(len(series_list)):
    r_centers = (list(series_list[i].bins)[0][0][1:]+list(series_list[i].bins)[0][0][:-1])/2.
    z_centers = (list(series_list[i].bins)[0][1][1:]+list(series_list[i].bins)[0][1][:-1])/2.
    r_mask = r_centers<(tpc.r-list(series_list[i].cut_dist)[0])
    z_mask = (z_centers>tpc.zmin+list(series_list[i].cut_dist)[0])\
          & (z_centers<tpc.zmax-list(series_list[i].cut_dist)[0])
    mask = np.transpose(np.tile(r_mask,(bins,1))) & np.tile(z_mask,(bins,1))
    rms = [np.sum((100*(element[mask]-h_true[mask])/h_true[mask])**2)/bins**2 for element in series_list[i].av_hist.values]
    #plt.loglog(series_list[i]['size'],rms,'.-',label=descrips[i])
plt.loglog([1e6,1e5,1e4,1e3],[0.01374239649879044,0.02197913281106599,0.08186015727434778,14.172411486552743],'.-',label=descrips[1])
plt.loglog([1e6,1e5,1e4,1e3],[1.1223136900849953,1.176265522540668,7.173270184392585,79.61007871215548],'.-',label=descrips[0])
plt.legend(loc='upper right')
plt.show()
plt.savefig(path+'rms_'+plot_name+'.png',bbox_inches='tight')

# plot comparison of different datasets
titles = ['1k events','10k events','100k events','1M events']
to_plot = ['a1k_final_0','a10k_final_0','a100k_final_0','a1M_final_0']
plot_data = data[[(name in to_plot) for name in data['name']]]
X,Y = np.meshgrid(np.linspace(rang[0][0],rang[0][1],bins),np.linspace(rang[1][1],rang[1][0],bins))
fig = plt.figure(figsize=(9, 6))
grid = ImageGrid(fig, 111,nrows_ncols=(1,len(titles)),axes_pad=0.10,share_all=True,\
                 cbar_location="right",cbar_mode="single",cbar_size="7%",cbar_pad=0.10)
eff = []
for i in range(len(grid)):
    hist = list(plot_data['av_hist'])[i]
    cut_dist = list(plot_data['cut_dist'])[i]
    err = np.transpose(np.flip(100*(hist-h_true)/h_true,axis=1))
    eff.append(np.ndarray.flatten(hist))
    h = np.transpose(np.flip(hist,axis=1))
    if error_plot==True:
        im = grid[i].imshow(err,cmap='RdBu_r',extent=[rang[0][0],rang[0][1],rang[1][0],rang[1][1]],vmin=-3,vmax=3)
        grid[i].contour(X,Y,abs(err),[1.],linestyles='dashed',linewidths=[0.5],colors=['black'])
        grid[i].contourf(X,Y,abs(err),[1.,100],colors=['black'],alpha=0.2)
        grid[i].axvspan(tpc.r-cut_dist,tpc.r,color='grey',lw=0.5,ls='-',zorder=10)
        grid[i].axhspan(tpc.zmin,tpc.zmin+cut_dist,color='grey',lw=0.5,ls='-',zorder=11)
        grid[i].axhspan(tpc.zmax-cut_dist,tpc.zmax,color='grey',lw=0.5,ls='-',zorder=12)
        cb_label = 'Percent Difference (Retrained vs Original NN)'
        f_name = 'errors'
    else:
        im = grid[i].imshow(h,cmap='Greys_r',extent=[rang[0][0],rang[0][1],rang[1][0],rang[1][1]],vmin=0.2,vmax=0.34)
        cb_label = 'Collection Efficiency'
        f_name = 'efficiency'
    grid[i].set_xlabel(r'$r$ (mm)')
    grid[i].set_xlim([0,tpc.r])
    grid[i].set_ylim([tpc.zmin,tpc.zmax])
    grid[i].set_title(titles[i])
    ax = plt.gca()
    if i>0:
        ax.set_yticks([])
grid[0].set_ylabel(r'$z$ (mm)')
cb = grid[-1].cax.colorbar(im)
grid[-1].cax.toggle_label(True)
cb.set_label_text(cb_label)
plt.savefig(path+f_name+'_'+plot_name+'.png',bbox_inches='tight')

# plot histogram of efficiency for various lightmaps
plt.figure(figsize=(5,4))
plt.xlabel('Collection Efficiency')
plt.ylabel('Probability Density')
plt.yscale('log')
plt.hist([np.ndarray.flatten(h_true)]+eff,label=['True lightmap']+titles,histtype=u'step',bins=100,density=True)
plt.legend(loc='upper left')
plt.savefig(path+'eff_hist_'+plot_name+'.png',bbox_inches='tight')

# plot histograms of error for different datasets
fig,axs = plt.subplots(1,len(titles),figsize=(9,4))
color = plt.cm.inferno(np.linspace(0, 1,len(titles)+1))
mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
for k in range(len(titles)):
    hist = list(plot_data['av_hist'])[k]
    err = np.transpose(np.flip(100*(hist-h_true)/h_true,axis=1))
    axs[k].hist(np.array(err).flatten(),bins=100,range=[-6,6],color='steelblue')
    axs[k].axvline(-1,linestyle='--',color='grey')
    axs[k].axvline(1,linestyle='--',color='grey')
    if k>0:
        axs[k].set_yticks([])
    else:
        ylim = np.array(axs[k].get_ylim())
        axs[0].set_ylabel('Counts')
    axs[k].set_ylim(ylim*3.5)
    axs[k].set_xlabel('Percent Error')
    axs[k].set_title(titles[k])
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.1,hspace=0.1)
plt.savefig(path+'error_hist_'+plot_name+'.png',bbox_inches='tight')

# plot histograms of the efficiency before training for Xe-127 and Rn-222 calibrations
e_gamma = np.loadtxt(path+'effic_100k_gamma_test.txt')
e_alpha = np.loadtxt(path+'effic_a100k_alpha_test.txt')
plt.figure(figsize=(5,4))
plt.xlabel('Collection Efficiency')
plt.ylabel('Probability Density')
#plt.yscale('log')
plt.hist([e_alpha,e_gamma],label=['Rn-222','Xe-127'],histtype=u'step',\
         color=['orangered','orange'],bins=100,density=True)
plt.legend(loc='upper left')
plt.savefig(path+'raw_eff_'+plot_name+'.png',bbox_inches='tight')
