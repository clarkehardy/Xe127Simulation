import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import histlite as hl
import LightMap
import sys
import os
import pickle
import copy
import argparse
import time
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic_2d

# ***********************************************************************************************************
# SET BASIC RUN OPTIONS HERE
# ***********************************************************************************************************

# save figures here
home_dir = os.getenv('WORKING_DIR')
data_dir = os.getenv('DATA_DIR')
path = '{}/outputs/'.format(data_dir)

# temporary way to get a subset of 100 events
smallset = False

'''
Arguments:
standoff     :    The fiducial cut as a distance from the nearest wall, in mm
description  :    The description that will be appended to file names to identify this run
-alphas      :    Choose Rn-222 calibration. Default is Xe-127 calibration
-testing     :    Run the script in testing mode. Default is False
-train       :    Run the full NN training. Default is False
-both_peaks  :    Train on both Xe-127 peaks. Default is just high energy peak. Not for use with -alphas
'''
parser = argparse.ArgumentParser()
parser.add_argument('-testing',action='store_true',default=False)
parser.add_argument('-train',action='store_true',default=False)
parser.add_argument('-alphas',action='store_true',default=False)
parser.add_argument('-both_peaks',action='store_true',default=False)
parser.add_argument('standoff',type=float)
parser.add_argument('description',type=str)
args = parser.parse_args()
testing = args.testing
rt_on = args.train
alphas = args.alphas
both_peaks = args.both_peaks
standoff = args.standoff
description = args.description

# simulation files to import
if testing==False and alphas==False:
    sets = [['{}/xe127_sims/TPCVessel_127Xe_1k.root'.format(data_dir)],\
            ['{}/xe127_sims/TPCVessel_127Xe_10k.root'.format(data_dir)],\
            ['{}/xe127_sims/TPCVessel_127Xe_100k_10.root'.format(data_dir)],\
            ['{}/xe127_sims/TPCVessel_127Xe_250k1.root'.format(data_dir),\
             '{}/xe127_sims/TPCVessel_127Xe_250k_1.root'.format(data_dir),\
             '{}/xe127_sims/TPCVessel_127Xe_250k_2.root'.format(data_dir),\
             '{}/xe127_sims/TPCVessel_127Xe_250k_3.root'.format(data_dir)]]
    name = ['1k_'+description,'10k_'+description,'100k_'+description,'1M_'+description]
elif testing==True and alphas==False:
    sets = [['{}/xe127_sims/TPCVessel_127Xe_100k_10.root'.format(data_dir)]]
    name = ['100k_'+description]
elif testing==True and alphas==True:
    sets = [['{}/rn222_sims/TPCVessel_222Rn_a50k_10.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_11.root'.format(data_dir)]]
    name = ['a100k_'+description]
else:
    sets = [['{}/rn222_sims/TPCVessel_222Rn_a50k_10.root'.format(data_dir)],\
            ['{}/rn222_sims/TPCVessel_222Rn_a50k_10.root'.format(data_dir)],\
            ['{}/rn222_sims/TPCVessel_222Rn_a50k_10.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_11.root'.format(data_dir)],\
            ['{}/rn222_sims/TPCVessel_222Rn_a50k_12.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_13.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_14.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_15.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_16.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_17.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_18.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_19.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_20.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_21.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_22.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_23.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_24.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_25.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_26.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_27.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_28.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_29.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_30.root'.format(data_dir),\
             '{}/rn222_sims/TPCVessel_222Rn_a50k_31.root'.format(data_dir)]]
    name = ['a1k_'+description,'a10k_'+description,'a100k_'+description,'a1M_'+description]

# **********************************************************************************************************
# DEFINE ANY FUNCTIONS HERE
# **********************************************************************************************************

# define a Gaussian
def gaus(x,A,mu,sigma):
    return A*np.exp(-(x-mu)**2/(2*sigma**2))

# cut to select high energy peak
cl_slope = -0.032
def peak_sep(x):
    return cl_slope*x+940

def cl_cut(x,y):
    if alphas==False:
        return (y>cl_slope*x+380) & (y<-x*cl_slope+400) & (y>-x*cl_slope-290)
    else:
        return y>25.*x-4e5
    
# *********************************************************************************************************
# SET PLOTTING OPTIONS AND DEFINE PLOTTING FUNCTIONS
# *********************************************************************************************************

# set plotting style
plt.rc('figure', dpi=200, figsize=(4,3), facecolor='w')
plt.rc('savefig', dpi=200, facecolor='w')
plt.rc('lines', linewidth=1.5)
pkw = dict(cmap='viridis',vmin=0., vmax=.5)

# function to plot LightMaps
def plot_lm_rz(ax, lm, theta=0, vectorize=False, cbar=True):
    f = lambda r, z: lm(r, (theta if vectorize else np.repeat(theta, z.size)), z, cyl=True)
    rang = (0, tpc.r), (tpc.zmin, tpc.zmax)
    h = hl.hist_from_eval(f, vectorize=vectorize, bins=200, range=rang)
    d = hl.plot2d(ax, h, cbar=cbar, **pkw)
    if cbar:
        d['colorbar'].set_label(r'Collection Efficiency')
    ax.set_xlabel(r'$r$ (mm)')
    ax.set_ylabel(r'$z$ (mm)')
    ax.set_aspect('equal')
    return d

# function to plot scatter plot with histograms
def proj2d(x,y,xlabel='x',ylabel='y',bins=200,s=0.001,color='blue'):
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.02
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    f = plt.figure(figsize=(6, 6))
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)
    ax_scatter.scatter(x, y, color=color, s=s, alpha=0.005)
    xmin = 0
    if alphas==False:
        ymin = 0
        xmax = 30000
        ymax = 30000
    else:
        ymin = 3.5e6
        xmax = 20000
        ymax = 4.5e6
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
    return ax_histx,ax_histy,ax_scatter

# *********************************************************************************************************
# LOAD TPC AND TRUE LIGHTMAP MODEL
# *********************************************************************************************************

# load TPC used for training
with open('{}/lm-analysis/tpc.pkl'.format(home_dir), 'rb') as handle:
    tpc = pickle.load(handle)
print(tpc)

# redefine TPC as reduced volume within field rings and between cathode and anode
# dimensions form preCDR
tpc.r = 566.65
tpc.zmax = tpc.zmax-19.
tpc.zmin = tpc.zmax-1183.

# load model
model_dir = '{}/lm-analysis/full-tpc'.format(home_dir)
lm_nn = LightMap.load_model(model_dir, 'LightMapNN')
print('\n', lm_nn, '\n')

# plot the original lightmap
fig,ax = plt.subplots(figsize=(3,5))
d = plot_lm_rz(ax,lm_nn)
ax.set_title('True Lightmap')
plt.savefig(path+'original.png',bbox_inches='tight')

# *********************************************************************************************************
# LOOP THROUGH ALL DATASETS
# *********************************************************************************************************

lim = [1e2,1e4] # normally [1e3,1e4]. Changed for testing
for i in range(len(name)):

    # *****************************************************************************************************
    # APPLY CUTS AND DETERMINE QUANTITIES FOR NN TRAINING
    # *****************************************************************************************************

    # import and read simulation files
    filenames = sets[i]
    data = LightMap.read_files(filenames,branches=['fInitNOP','fNTE','fTotalEventEnergy','fNESTLineageX','fNESTLineageY','fNESTLineageZ','fNESTLineageNTE','fNESTLineageNOP'])
    if (alphas==True and (i<2 and testing==False)) or smallset==True:
        data = data[:int(lim[i])]
    minZ = data.z.min()
    data['z'] = data.z.values + (tpc.zmin - minZ)
    print(data.z.min())
    print(data.z.max())
    print(max(np.sqrt(data.x.values**2+data.y.values**2)))
    data['fNESTLineageZ'] = data['fNESTLineageZ'] + (tpc.zmin - minZ)
    print(data.head())

    # detected thermal electrons is sum of thermal electrons produced in lineages within field cage
    start_time = time.time()
    fNTEDetected = []
    xWeighted = []
    yWeighted = []
    zWeighted = []
    xOP = []
    yOP = []
    zOP = []
    for index, evt in data.iterrows():
        index_array = (evt['fNESTLineageX']**2+evt['fNESTLineageY']**2 < tpc.r**2) & ((evt['fNESTLineageZ'] > tpc.zmin) & (evt['fNESTLineageZ'] < tpc.zmax))
        fNTEDetected.append(np.sum(evt['fNESTLineageNTE'][index_array]))
        xWeighted.append(np.sum(evt['fNESTLineageNTE'][index_array]*evt['fNESTLineageX'][index_array])/np.sum(evt['fNESTLineageNTE'][index_array]))
        yWeighted.append(np.sum(evt['fNESTLineageNTE'][index_array]*evt['fNESTLineageY'][index_array])/np.sum(evt['fNESTLineageNTE'][index_array]))
        zWeighted.append(np.sum(evt['fNESTLineageNTE'][index_array]*evt['fNESTLineageZ'][index_array])/np.sum(evt['fNESTLineageNTE'][index_array]))
        xOP.append(np.sum(evt['fNESTLineageNOP']*evt['fNESTLineageX'])/np.sum(evt['fNESTLineageNOP']))
        yOP.append(np.sum(evt['fNESTLineageNOP']*evt['fNESTLineageY'])/np.sum(evt['fNESTLineageNOP']))
        zOP.append(np.sum(evt['fNESTLineageNOP']*evt['fNESTLineageZ'])/np.sum(evt['fNESTLineageNOP']))
        
    elapsed_time = time.time()-start_time
    print('Time elapsed: '+str(elapsed_time))

    # create new columns in the dataframe for corrected positions and NTE
    data['xWeighted'] = xWeighted
    data['yWeighted'] = yWeighted
    data['zWeighted'] = zWeighted
    data['xOP'] = xOP
    data['yOP'] = yOP
    data['zOP'] = zOP
    data['rOP'] = np.sqrt(np.array(xOP)**2+np.array(yOP)**2)
    data['fNTEDetected'] = fNTEDetected
    
    # cut events with no charge signal
    data_size = len(data.index)
    cuts = ~(data['fNTEDetected']==0)
    after_elec = len(data[cuts].index)
    
    # cut events with no photons produced
    cuts = ~(data['fInitNOP']==0)
    after_photon = len(data[cuts].index)
    
    # apply fiducial cut
    fidcut = True
    if(fidcut==True):
        zlim = [tpc.zmin+standoff,tpc.zmax-standoff]
        rlim = [0,tpc.r-standoff]
    else:
        zlim = [tpc.zmin,tpc.zmax]
        rlim = [0,tpc.r]
    inside_z = abs(data.zWeighted.values-(zlim[1]-zlim[0])/2.-zlim[0])>(zlim[1]-zlim[0])/2.
    inside_r = abs(data.rOP.values-(rlim[1]-rlim[0])/2.-rlim[0])>(rlim[1]-rlim[0])/2.
    cuts = cuts & (~inside_z & ~inside_r)
    after_fiducial = len(data[cuts].index)

    # sample based on number of photons generated
    qe = 0.1
    data['ndet'] = lm_nn.sample_n_collected(data.xOP.values, data.yOP.values, data.zOP.values, data.fInitNOP.values, qe=qe, ap=0.2, seed=1)
    print('\nDetected photons sampled.')
    print(np.sum(data['ndet']==0))
    
    # smear electron signal
    nelec = np.array(data['fNTEDetected'])
    charge_bins = np.linspace(0,28000,201)
    charge_fluc = 2200
    data['fNTEDetected'] = data.fNTEDetected.values+np.random.normal(scale=charge_fluc,size=len(data['fNTEDetected']))
    data.loc[data.fNTEDetected.values<0,'fNTEDetected'] = 0

    # separate low and high energy peaks
    data['peak'] = np.ones(len(data.ndet.values))
    if alphas==False:
        peak_cond = peak_sep(data.fNTEDetected.values) < data.ndet.values
        data.loc[peak_cond,'peak'] = 2

    # cut out data that is not in one of the peaks
    cut_cond = cl_cut(data.fNTEDetected.values,data.ndet.values)
    cuts = cuts & cut_cond
    after_chargelight = len(data[cuts].index)

    # print results of cuts with efficiency
    print('\nEvents before thermal electron cut: '+str(data_size))
    print('Events after thermal electron cut: '+str(after_elec))
    print('Thermal electron cut efficiency: {:.1f} %'.format(after_elec*100./data_size))
    print('Events after photon cut: '+str(after_photon))
    print('Photon cut efficiency: {:.1f} %'.format(after_photon*100./after_elec))
    print('Events after fiducial cut: '+str(after_fiducial))
    print('Fiducial cut efficiency: {:.1f} %'.format(after_fiducial*100./after_photon))
    print('Events after charge/light cut: '+str(after_chargelight))
    print('Charge/light cut efficiency: {:.1f} %\n'.format(after_chargelight*100./after_fiducial))

    # fit Gaussian to high energy peak
    fNTE_counts,fNTE_bins = np.histogram(data.fNTEDetected.values[data['peak']==2],bins=charge_bins)
    fNTE_bins = (fNTE_bins[1:]+fNTE_bins[:-1])/2.
    popt_gaus,pcov_gaus = curve_fit(gaus,fNTE_bins,fNTE_counts,p0=[max(fNTE_counts),18000,2000])
    print('Width of smeared charge signal is '+str(np.around(100*popt_gaus[2]/popt_gaus[1],decimals=1))+' %\n')

    # correct for efficiency with true Lightmap
    r_vals = np.array(data.rOP.values)
    theta_vals = np.arctan(data.yOP.values/data.xOP.values)
    ndet_effic = lm_nn(r_vals,theta_vals,data.zOP.values,cyl=True)

    # compute mean number of photons for each peak
    if alphas==False:
        peaks = np.array((0,0))
        for j in range(2):
            peaks[j] = np.mean(data['fInitNOP'][(data['peak']==j+1) & cuts])
    else:
        peaks = [np.mean(data['fInitNOP'][cuts])]

    # compute efficiency for each peak separately
    if alphas==False:
        data['eff'] = data.ndet.values/(qe*peaks[np.array(data.peak.values-1,dtype=int)])
    else:
        data['eff'] = data.ndet.values/(qe*peaks[0])
    
    np.savetxt(path+'effic_'+name[i]+'.txt',np.array(data.eff.values))

    # *****************************************************************************************************
    # PLOT ALL DATA BEFORE TRAINING
    # *****************************************************************************************************

    # plot spatial distribution of events
    from matplotlib.image import NonUniformImage
    fig,ax = plt.subplots(figsize=(3,5))
    hist,r_edges,z_edges = np.histogram2d(data.rOP.values,data.zOP.values,bins=50)
    r_bins = (r_edges[:-1]+r_edges[1:])/2.
    z_bins = (z_edges[:-1]+z_edges[1:])/2.
    R,Z = np.meshgrid(r_bins,z_bins)
    areas = np.pi*(r_edges[1:]**2-r_edges[:-1]**2)
    Areas,Zbins = np.meshgrid(areas,z_bins)
    im = ax.pcolormesh(R,Z,hist.transpose()/Areas)
    fig.colorbar(im)
    ax.set_xlim([0,tpc.r])
    ax.set_ylim([tpc.zmin,tpc.zmax])
    ax.set_xlabel(r'$r$ (mm)')
    ax.set_ylabel(r'$z$ (mm)')
    ax.set_aspect('equal')
    plt.savefig(path+'spatial_rz_'+name[i]+'.png',bbox_inches='tight')

    # plot raw scatter colored by efficiency
    plt.figure(figsize=(3,5))
    plt.scatter(data.rOP.values[cuts],data.zOP.values[cuts],c=data.eff.values[cuts],s=0.1,cmap='spring')
    plt.xlabel(r'$r$ (mm)')
    plt.ylabel(r'$z$ (mm)')
    plt.title('Calibration Data')
    cbar = plt.colorbar()
    cbar.set_label('Collection Efficiency')
    ax = plt.gca()
    ax.set_xlim([0,tpc.r])
    ax.set_ylim([tpc.zmin,tpc.zmax])
    ax.set_aspect('equal')
    plt.savefig(path+'raw_scatter_'+name[i]+'.png',bbox_inches='tight')

    # plot spatial distribution x and y
    plt.figure(figsize=(3,3))
    hist,x_edges,y_edges = np.histogram2d(data.xOP.values,data.yOP.values,bins=200)
    x_bins = (x_edges[:-1]+x_edges[1:])/2.
    y_bins = (y_edges[:-1]+y_edges[1:])/2.
    X,Y = np.meshgrid(x_bins,y_bins)
    plt.pcolormesh(X,Y,hist)
    plt.colorbar()
    plt.xlabel(r'$x$ (mm)')
    plt.ylabel(r'$y$ (mm)')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.savefig(path+'spatial_xy_'+name[i]+'.png',bbox_inches='tight')

    # plot fiducial cut
    fig,ax = plt.subplots(figsize=(3,5))
    d = plot_lm_rz(ax,lm_nn)
    plt.axvline(x=rlim[0],ymin=(zlim[0]-tpc.zmin)/(tpc.zmax-tpc.zmin),\
                ymax=(zlim[1]-tpc.zmin)/(tpc.zmax-tpc.zmin),color='red',ls='--',lw=0.5)
    plt.axvline(x=rlim[1],ymin=(zlim[0]-tpc.zmin)/(tpc.zmax-tpc.zmin),\
                ymax=(zlim[1]-tpc.zmin)/(tpc.zmax-tpc.zmin),color='red',ls='--',lw=0.5)
    plt.axhline(y=zlim[0],xmin=rlim[0]/tpc.r,xmax=rlim[1]/tpc.r,color='red',ls='--',lw=0.5)
    plt.axhline(y=zlim[1],xmin=rlim[0]/tpc.r,xmax=rlim[1]/tpc.r,color='red',ls='--',lw=0.5)
    plt.xlim([0,tpc.r])
    plt.ylim([tpc.zmin,tpc.zmax])
    ax.set_title('Fiducial Volume Cut')
    plt.savefig(path+'fid_cut_'+name[i]+'.png',bbox_inches='tight')

    # plot energy spectrum
    plt.figure()
    plt.hist(data.fTotalEventEnergy.values,bins=200,color='blue',histtype=u'step')
    plt.yscale('log')
    plt.xlabel('Total Energy (MeV)')
    plt.ylabel('Counts')
    plt.savefig(path+'spectrum_'+name[i]+'.png',bbox_inches='tight')

    # plot MC truth scatter plot
    ax_histx,ax_histy,ax_scatter = proj2d(nelec,data.fInitNOP.values,xlabel='MC Truth Ionization (Number of Electrons)',\
                                          ylabel='MC Truth Scintillation (Number of Photons)',s=0.2)
    plt.savefig(path+'MCTruth_'+name[i]+'.png',bbox_inches='tight')

    # plot charge and light with selection cut
    xvals = np.linspace(0,26500,10)
    yvals = peak_sep(xvals)
    plt.figure()
    plt.hist2d(data.fNTEDetected.values[cuts],data.ndet.values[cuts],200,norm=mpl.colors.LogNorm())
    plt.xlabel('Detected electrons')
    plt.ylabel('Detected photons')
    if alphas==False:
        plt.plot(xvals,yvals,'--r',lw=1)
        plt.xlim([0,30000])
        plt.ylim([0,900])
    else:
        plt.plot(np.linspace(0,40000,1000),25.*np.linspace(0,40000,1000)-4e5,'--r')
    plt.colorbar()
    plt.savefig(path+'charge_light_'+name[i]+'.png',bbox_inches='tight')

    # plot a histogram of the charge signal
    plt.figure()
    plt.hist([nelec,data.fNTEDetected.values],bins=charge_bins,color=['black','blue'],label=['MC Truth charge','Observed charge'],histtype=u'step')
    plt.plot(np.linspace(12000,25000,100),gaus(np.linspace(12000,25000,100),*popt_gaus),color='pink')
    plt.xlabel('Charge Signal (Number of Electrons)')
    plt.ylabel('Counts')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(path+'charge_'+name[i]+'.png')
    
    # plot a 2d histogram of the light signal
    plt.figure()
    plt.hist2d(data.fInitNOP.values,data.ndet.values,bins=200,norm=mpl.colors.LogNorm())
    plt.xlabel('MC Truth Scintillation (Number of Photons)')
    plt.ylabel('Detected Photoelectrons')
    plt.colorbar()
    plt.savefig(path+'light_'+name[i]+'.png',bbox_inches='tight')

    # plot a histogram of the light signal
    '''
    plt.figure()
    plt.hist([data.fInitNOP.values[data['peak']==1]/np.mean(data.fInitNOP.values[data['peak']==1]),data.ndet.values[data['peak']==1]/np.mean(data.ndet.values[data['peak']==1])],\
             bins=200,color=['darkorange','red'],histtype=u'step',density=False,label=['MC Truth','Detected'])
    plt.xlabel('Relative Number of Photons')
    plt.ylabel('Counts')
    plt.title('Detected Photon Sampling')
    plt.legend(loc='best')
    plt.savefig(path+'lighthist_'+name[i]+'.png',bbox_inches='tight')
    '''
    
    # 2d histogram of reconstructed light produced
    '''
    plt.figure()
    plt.hist2d(data.fInitNOP.values,ndet_corr,bins=200,norm=mpl.colors.LogNorm())
    plt.xlabel('MC Truth Scintillation (Number of Photons)')
    plt.ylabel('Reconstructed Scintillation (Number of Photons)')
    plt.colorbar()
    plt.savefig(path+'light_recon_'+name[i]+'.png',bbox_inches='tight')
    '''

    # plot efficiency curve for both peaks
    plt.figure()
    plt.hist([data['eff'][data['peak']==1],data['eff'][data['peak']==2],data['eff']],bins=200,color=['green','blue','black'],\
             histtype=u'step',label=['Low E peak','High E peak','Both peaks'])
    plt.xlabel('Photon Collection Efficiency')
    plt.ylabel('Counts')
    if alphas==False:
        plt.legend(loc='upper right')
    plt.savefig(path+'eff_compare_'+name[i]+'.png',bbox_inches='tight')

    plt.show()
    plt.close('all')
    if(rt_on == False):
        continue

    # *****************************************************************************************************
    # TRAIN NN ON SIMULATED DATA
    # *****************************************************************************************************

    # define new training set
    if both_peaks == True:
        #train_again = data.x.values[cuts], data.y.values[cuts], data.z.values[cuts], data.eff.values[cuts]
        train_again = data.xOP.values[cuts], data.yOP.values[cuts], data.zOP.values[cuts], data.eff.values[cuts]
    else:
        #train_again = data.x.values[(data['peak']==2) & cuts],data.y.values[(data['peak']==2) & cuts],\
        #              data.z.values[(data['peak']==2) & cuts],data.eff.values[(data['peak']==2) & cuts]
        train_again = data.xOP.values[(data['peak']==2) & cuts],data.yOP.values[(data['peak']==2) & cuts],\
                      data.zOP.values[(data['peak']==2) & cuts],data.eff.values[(data['peak']==2) & cuts]
    layers = [512, 256, 128, 64, 32]
    lm_nn_again = LightMap.total.LightMapNN(tpc, epochs=10, batch_size=64, hidden_layers=layers)

    # train new set
    for j in range(3):
        lm_nn_again.fit(*train_again)
    print(lm_nn_again)

    # save lightmap neural net
    LightMap.save_model(path+'LightMap_'+name[i],lm_nn_again.kind,lm_nn_again)

    # *****************************************************************************************************
    # PLOT FINAL LIGHTMAP FOR SIMULATED DATA
    # *****************************************************************************************************

    # plot differences between LightMaps
    fig, ax = plt.subplots()
    rang = (0, tpc.r), (tpc.zmin, tpc.zmax)
    f = lambda r, z: lm_nn(r, np.repeat(0, r.size), z, cyl=True)
    h_nn = hl.hist_from_eval(f, vectorize=False, bins=200, range=rang)
    f = lambda r, z: lm_nn_again(r, np.repeat(0, r.size), z, cyl=True)
    h_nn_again = hl.hist_from_eval(f, vectorize=False, bins=200, range=rang)
    d = hl.plot2d(ax, 100*(h_nn_again - h_nn) / h_nn, cbar=True, cmap='RdBu_r',vmin=-10,vmax=10)
    d['colorbar'].set_label(r'Percent Difference (Retrained vs Original NN)')
    ax.set_xlabel(r'$r$ (mm)')
    ax.set_ylabel(r'$z$ (mm)')
    ax.set_aspect('equal')
    plt.savefig(path+'difference_'+name[i]+'.png',bbox_inches='tight')

    # plot side-by-side of original and retrained LightMap
    fig, axs = plt.subplots(1, 3, figsize=(4.5,3), gridspec_kw=dict(width_ratios=[1,1,.05]))
    these_lms = lm_nn, lm_nn_again
    titles = 'Original NN', 'Round-trip NN',
    for (ax,lm, title) in zip(axs, these_lms, titles):
        d = plot_lm_rz(ax, lm, cbar=False)
        ax.set_title(title)
    cb = fig.colorbar(d, cax=axs[-1])
    cb.set_label(r'Collection Efficiency')
    for ax in axs[1:-1]:
        ax.set_yticks([])
        ax.set_ylabel('')
    plt.savefig(path+'compare_'+name[i]+'.png',bbox_inches='tight')
    plt.show()
    plt.close('all')
    del lm_nn_again
