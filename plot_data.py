import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import histlite as hl

# set plotting style
plt.rc('figure', dpi=200, figsize=(4,3), facecolor='w')
plt.rc('savefig', dpi=200, facecolor='w')
plt.rc('lines', linewidth=1.5)
pkw = dict(cmap='viridis',vmin=0., vmax=.5)

# function to plot LightMaps
def plot_lm_rz(ax, lm, tpc, theta=0, vectorize=False, cbar=True):
    f = lambda r, z: lm(r, (theta if vectorize else np.repeat(theta, z.size)), z, cyl=True)
    rang = (0, tpc.r), (tpc.zmin, tpc.zmax)
    h = hl.hist_from_eval(f, vectorize=vectorize, bins=1000, range=rang)
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
    f = plt.figure(figsize=(5, 5))
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)
    ax_scatter.scatter(x, y, color=color, s=s, alpha=0.005)
    xmin = 0
    ymin = 0
    xmax = 30000
    ymax = 30000
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

def make_figs(tpc,lm_true,data,cuts,path,name,rlim,zlim,peak_sep):
    # plot spatial distribution of events in r and z
    fig,ax = plt.subplots(figsize=(3.5,5))
    hist,r_edges,z_edges = np.histogram2d(data.weighted_radius.values[cuts],data.z.values[cuts],bins=50,range=([rlim,zlim]),density=True)
    r_bins = (r_edges[:-1]+r_edges[1:])/2.
    z_bins = (z_edges[:-1]+z_edges[1:])/2.
    R,Z = np.meshgrid(r_bins,z_bins)
    im = ax.imshow(hist.T/(2*np.pi*r_bins[np.newaxis,:]),extent=[r_edges.min(),r_edges.max(),z_edges.min(),z_edges.max()])
    vol = rlim[1]**2*np.pi*(zlim[1]-zlim[0])
    cbar = plt.colorbar(im)
    cbar.set_label('Event density (events/mm$^3$)')
    ax.set_xlim([0,tpc.r])
    ax.set_ylim([tpc.zmin,tpc.zmax])
    ax.set_xlabel(r'$r$ (mm)')
    ax.set_ylabel(r'$z$ (mm)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(path+'spatial_rz_'+name+'.png',bbox_inches='tight')

    # plot spatial distribution x and y
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    hist,x_edges,y_edges = np.histogram2d(data.weighted_x.values[cuts],data.weighted_y.values[cuts],bins=200,density=True)
    x_bins = (x_edges[:-1]+x_edges[1:])/2.
    y_bins = (y_edges[:-1]+y_edges[1:])/2.
    im = ax.imshow(hist.T/(zlim[1]-zlim[0]),extent=[x_edges.min(),x_edges.max(),y_edges.min(),y_edges.max()])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(im,cax=cax)
    cbar.set_label('Event density (events/mm$^3$)')
    ax.set_xlabel(r'$x$ (mm)')
    ax.set_ylabel(r'$y$ (mm)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(path+'spatial_xy_'+name+'.png',bbox_inches='tight')

    # plot raw scatter colored by efficiency
    plt.figure(figsize=(3.5,5))
    plt.scatter(data.weighted_radius.values[cuts],data.z.values[cuts],c=data.eff.values[cuts],s=0.1,cmap='spring')
    plt.xlabel(r'$r$ (mm)')
    plt.ylabel(r'$z$ (mm)')
    plt.title('Calibration Data')
    cbar = plt.colorbar()
    ax = plt.gca()
    cbar.set_label('Collection Efficiency')
    ax.set_xlim([0,tpc.r])
    ax.set_ylim([tpc.zmin,tpc.zmax])
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(path+'raw_scatter_'+name+'.png',bbox_inches='tight')

    # plot fiducial cut
    fig,ax = plt.subplots(figsize=(3.5,5))
    d = plot_lm_rz(ax,lm_true,tpc)
    plt.axvline(x=rlim[0],ymin=(zlim[0]-tpc.zmin)/(tpc.zmax-tpc.zmin),\
                ymax=(zlim[1]-tpc.zmin)/(tpc.zmax-tpc.zmin),color='red',ls='--',lw=0.5)
    plt.axvline(x=rlim[1],ymin=(zlim[0]-tpc.zmin)/(tpc.zmax-tpc.zmin),\
                ymax=(zlim[1]-tpc.zmin)/(tpc.zmax-tpc.zmin),color='red',ls='--',lw=0.5)
    plt.axhline(y=zlim[0],xmin=rlim[0]/tpc.r,xmax=rlim[1]/tpc.r,color='red',ls='--',lw=0.5)
    plt.axhline(y=zlim[1],xmin=rlim[0]/tpc.r,xmax=rlim[1]/tpc.r,color='red',ls='--',lw=0.5)
    ax.set_xlim([0,tpc.r])
    ax.set_ylim([tpc.zmin,tpc.zmax])
    ax.set_title('Fiducial Volume Cut')
    plt.tight_layout()
    plt.savefig(path+'fid_cut_'+name+'.png',bbox_inches='tight')

    # plot MC truth scatter plot
    ax_histx,ax_histy,ax_scatter = proj2d(data.fNTE.values,data.fInitNOP.values,xlabel='MC Truth Ionization (Number of Electrons)',\
                                          ylabel='MC Truth Scintillation (Number of Photons)',s=0.2)
    plt.savefig(path+'MCTruth_'+name+'.png',bbox_inches='tight')

    # plot charge and light with selection cut
    xvals = np.linspace(0,26500,10)
    yvals = peak_sep(xvals)
    plt.figure(figsize=(4,3))
    plt.hist2d(data.evt_charge_including_noise.values[cuts],data['Observed Light'][cuts],200,norm=mpl.colors.LogNorm())
    plt.xlabel('Detected electrons')
    plt.ylabel('Detected photons')
    plt.plot(xvals,yvals,'--r',lw=1)
    plt.xlim([0,30000])
    plt.ylim([0,900])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path+'charge_light_'+name+'.png',bbox_inches='tight')

    # plot a histogram of the charge signal
    plt.figure(figsize=(4,3))
    plt.hist(data.evt_charge_including_noise.values[cuts],bins=100,color='blue',label='Observed charge',histtype=u'step')
    plt.xlabel('Charge Signal (Number of Electrons)')
    plt.ylabel('Counts')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(path+'charge_'+name+'.png')

    # plot a 2d histogram of the light signal
    plt.figure(figsize=(4,3))
    plt.hist2d(data.fInitNOP.values[cuts],data['Observed Light'][cuts],bins=100,norm=mpl.colors.LogNorm())
    plt.xlabel('MC Truth Scintillation (Number of Photons)')
    plt.ylabel('Detected Photoelectrons')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path+'light_'+name+'.png',bbox_inches='tight')

    # plot a histogram of the light signal
    plt.figure(figsize=(4,3))
    plt.hist([data.fInitNOP.values[data['peak']==1]/np.mean(data.fInitNOP.values[data['peak']==1]),\
              data['Observed Light'][data['peak']==1]/np.mean(data['Observed Light'][data['peak']==1])],\
             bins=100,color=['darkorange','red'],histtype=u'step',density=False,label=['MC Truth','Detected'])
    plt.xlabel('Relative Number of Photons')
    plt.ylabel('Counts')
    plt.title('Detected Photon Sampling')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(path+'lighthist_'+name+'.png',bbox_inches='tight')

    # plot efficiency curve for both peaks
    plt.figure(figsize=(4,3))
    plt.hist([data['eff'][data['peak']==1],data['eff'][data['peak']==2],data['eff']],bins=100,\
             color=['green','blue','black'],histtype=u'step',label=['Low E peak','High E peak','Both peaks'])
    plt.xlabel('Photon Collection Efficiency')
    plt.ylabel('Counts')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(path+'eff_compare_'+name+'.png',bbox_inches='tight')

def plot_results(tpc,lm_true,lm_again,rlim,zlim,path,name):
    # plot differences between LightMaps
    fig, ax = plt.subplots(1,1,figsize=(4,5))
    rang = (rlim[0],rlim[1]), (zlim[0],zlim[1])
    f = lambda r, z: lm_true(r, np.repeat(0, r.size), z, cyl=True)
    h_nn = hl.hist_from_eval(f, vectorize=False, bins=200, range=rang)
    f = lambda r, z: lm_again(r, np.repeat(0, r.size), z, cyl=True)
    h_again = hl.hist_from_eval(f, vectorize=False, bins=200, range=rang)
    d = hl.plot2d(ax, h_nn / h_again, cbar=True, cmap='RdBu_r',vmin=0.95,vmax=1.05)
    d['colorbar'].set_label(r'True Lightmap / Corrected Lightmap')
    ax.set_xlabel(r'$r$ (mm)')
    ax.set_ylabel(r'$z$ (mm)')
    ax.set_title(r'Corrected Lightmap Accuracy')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(path+'difference_'+name+'.png',bbox_inches='tight')

    # plot side-by-side of original and retrained LightMap
    fig, axs = plt.subplots(1, 3, figsize=(6,5), gridspec_kw=dict(width_ratios=[1,1,.05]))
    these_lms = lm_true, lm_again
    titles = 'Original Lightmap', 'Round-trip Lightmap',
    for (ax,lm, title) in zip(axs, these_lms, titles):
        d = plot_lm_rz(ax, lm, tpc, cbar=False)
        ax.set_title(title)
    cb = fig.colorbar(d, cax=axs[-1])
    cb.set_label(r'Collection Efficiency')
    for ax in axs[1:-1]:
        ax.set_yticks([])
        ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(path+'compare_'+name+'.png',bbox_inches='tight')

    # plot histogram of the reconstructed lightmap accuracy
    var = np.var(h_nn.values/h_again.values)
    mean = np.mean(h_nn.values/h_again.values)
    plt.figure(figsize=(4,3))
    acc = np.ndarray.flatten(h_nn.values/h_again.values)
    plt.hist(acc,bins=100,range=(max([0.8,min(acc)]),min([1.2,max(acc)])),color='navy',histtype=u'step')
    plt.title('Corrected Lightmap Accuracy')
    plt.xlabel('True Lightmap / Corrected Lightmap')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(path+'accuracy_'+name+'.png',bbox_inches='tight')

    return mean,var
