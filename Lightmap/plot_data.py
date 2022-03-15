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
        d['colorbar'].set_label(r'Photon Transport Efficiency')
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
    xmax = 25000
    ymax = 25000
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

def make_figs(tpc,lm_true,data,cuts,path,name,rlim,zlim,peak_sep):
    # plot spatial distribution of events in r and z
    fig,ax = plt.subplots(figsize=(3.5,5))
    hist,r_edges,z_edges = np.histogram2d(data.weighted_radius.values[cuts],data.z.values[cuts],bins=50,range=([rlim,zlim]),density=True)
    r_bins = (r_edges[:-1]+r_edges[1:])/2.
    z_bins = (z_edges[:-1]+z_edges[1:])/2.
    im = ax.imshow(hist.T*np.sum(cuts)/(2*np.pi*r_bins[np.newaxis,:]),extent=[r_edges.min(),r_edges.max(),z_edges.min(),z_edges.max()])
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
    fig.savefig(path+'spatial_rz_'+name+'.png',bbox_inches='tight')

    # plot spatial distribution x and y
    fig,ax = plt.subplots(figsize=(5,5))
    hist,x_edges,y_edges = np.histogram2d(data.weighted_x.values[cuts],data.weighted_y.values[cuts],bins=200,density=True)
    x_bins = (x_edges[:-1]+x_edges[1:])/2.
    y_bins = (y_edges[:-1]+y_edges[1:])/2.
    im = ax.imshow(hist.T*np.sum(cuts)/(zlim[1]-zlim[0]),extent=[x_edges.min(),x_edges.max(),y_edges.min(),y_edges.max()])
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
    fig.savefig(path+'spatial_xy_'+name+'.png',bbox_inches='tight')

    # plot raw scatter colored by efficiency
    fig,ax = plt.subplots(figsize=(3.5,5))
    sc = ax.scatter(data.weighted_radius.values[cuts],\
                    data.z.values[cuts],c=data.eff.values[cuts],\
                    s=0.1,cmap='spring',vmin=0.2,vmax=0.4)
    ax.set_xlabel(r'$r$ (mm)')
    ax.set_ylabel(r'$z$ (mm)')
    ax.set_title('Calibration Events')
    cbar = fig.colorbar(sc,format='%.2f',ticks=[0.2,0.25,0.3,0.35,0.4])
    cbar.set_label('Photon Transport Efficiency')
    ax.set_xlim([0,tpc.r])
    ax.set_ylim([tpc.zmin,tpc.zmax])
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(path+'raw_scatter_'+name+'.png',bbox_inches='tight')

    # plot fiducial cut
    fig,ax = plt.subplots(figsize=(3.5,5))
    d = plot_lm_rz(ax,lm_true,tpc)
    ax.axvline(x=rlim[0],ymin=(zlim[0]-tpc.zmin)/(tpc.zmax-tpc.zmin),\
                ymax=(zlim[1]-tpc.zmin)/(tpc.zmax-tpc.zmin),color='red',ls='--',lw=0.5)
    ax.axvline(x=rlim[1],ymin=(zlim[0]-tpc.zmin)/(tpc.zmax-tpc.zmin),\
                ymax=(zlim[1]-tpc.zmin)/(tpc.zmax-tpc.zmin),color='red',ls='--',lw=0.5)
    ax.axhline(y=zlim[0],xmin=rlim[0]/tpc.r,xmax=rlim[1]/tpc.r,color='red',ls='--',lw=0.5)
    ax.axhline(y=zlim[1],xmin=rlim[0]/tpc.r,xmax=rlim[1]/tpc.r,color='red',ls='--',lw=0.5)
    ax.set_xlim([0,tpc.r])
    ax.set_ylim([tpc.zmin,tpc.zmax])
    ax.set_title('Fiducial Volume Cut')
    fig.tight_layout()
    fig.savefig(path+'fid_cut_'+name+'.png',bbox_inches='tight')

    # plot MC truth scatter plot
    fig,ax_histx,ax_histy,ax_scatter = proj2d(data.fNTE.values[cuts],data.fInitNOP.values[cuts],xlabel='MC Truth Ionization (Number of Electrons)',\
                                          ylabel='MC Truth Scintillation (Number of Photons)',s=0.2)
    fig.savefig(path+'MCTruth_'+name+'.png',bbox_inches='tight')

    # plot charge and light with selection cut
    xvals = np.linspace(0,26500,10)
    yvals = peak_sep(xvals)
    plt.rcParams.update({'font.size': 18})
    fig,ax = plt.subplots(figsize=(8,6))
    ht = ax.hist2d(data.evt_charge_including_noise.values[cuts],data['Observed Light'][cuts],200,norm=mpl.colors.LogNorm())
    ax.set_xlabel('Detected Electrons')
    ax.set_ylabel('Detected Photons')
    ax.set_xlim([8000,28000])
    ax.set_ylim([200,1700])
    ax.plot(xvals,yvals,'--r',lw=4)
    cbar = fig.colorbar(ht[3])
    fig.tight_layout()
    fig.savefig(path+'charge_light_'+name+'.png',bbox_inches='tight')
    plt.rcParams.update({'font.size': 10})

    # plot a histogram of the charge signal
    fig,ax = plt.subplots(figsize=(4,3))
    ax.hist([data['fNTE'][cuts],data.evt_charge_including_noise.values[cuts]],bins=100,\
            color=['blue','green'],label=['MC truth electrons','Detected electrons'],histtype=u'step')
    ax.set_xlabel('Number of Electrons')
    ax.set_ylabel('Counts')
    ax.legend(loc='best',prop={'size': 8})
    fig.tight_layout()
    fig.savefig(path+'charge_'+name+'.png')

    # plot a 2d histogram of the light signal
    fig,ax = plt.subplots(figsize=(4,3))
    ht = ax.hist2d(data.fInitNOP.values[cuts],data['Observed Light'][cuts],bins=200,norm=mpl.colors.LogNorm())
    ax.set_xlabel('MC Truth Photons')
    ax.set_ylabel('Detected Photons')
    fig.colorbar(ht[3])
    fig.tight_layout()
    fig.savefig(path+'light_'+name+'.png',bbox_inches='tight')

    # plot a histogram of the light signal
    plt.rcParams.update({'font.size': 18})
    fig,ax = plt.subplots(figsize=(8,6))
    truth_hist,truth_bins = np.histogram(data.fInitNOP.values[cuts],bins=100)
    det_hist,det_bins = np.histogram(data['Observed Light'][cuts],bins=100)
    truth_bins = (truth_bins[1:]+truth_bins[:-1])/2.
    det_bins = (det_bins[1:]+det_bins[:-1])/2.
    scale = truth_bins[np.argmax(truth_hist)]/det_bins[np.argmax(det_hist)]
    ax.hist([data.fInitNOP.values[cuts],data['Observed Light'][cuts]*scale,\
             data['Observed Light'][cuts]/0.1/lm_true(data.weighted_x.values[cuts],data.weighted_y.values[cuts],data.z.values[cuts])],\
            bins=100,range=(0,30000),color=['darkorange','red','maroon'],histtype=u'step',density=False,\
            label=['MC truth','No lightmap','Perfect lightmap'])
    ax.set_xlabel('Scintillation (photons)')
    ax.set_ylabel('Counts (A.U.)')
    ax.set_xlim([0,30000])
    y_vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.0f}'.format(x * 1e-4) for x in y_vals])
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(path+'lighthist_'+name+'.png',bbox_inches='tight')
    plt.rcParams.update({'font.size': 10})

    # plot efficiency curve for both peaks
    fig,ax = plt.subplots(figsize=(4,3))
    ax.hist([data['eff'][cuts & (data['peak']==1)],data['eff'][cuts & (data['peak']==2)],data['eff'][cuts]],bins=100,\
             color=['green','blue','black'],histtype=u'step',label=['Low E peak','High E peak','Both peaks'])
    ax.set_xlabel('Photon Collection Efficiency')
    ax.set_ylabel('Counts')
    ax.legend(loc='best',prop={'size': 8})
    fig.tight_layout()
    fig.savefig(path+'eff_compare_'+name+'.png',bbox_inches='tight')

def plot_results(tpc,lm_true,lm_again,rlim,zlim,path,name):
    # compute lightmap on a grid uniformly by volume for accuracy calculations
    rang_uniform = (rlim[0]**2,rlim[1]**2), (zlim[0],zlim[1])
    f = lambda r2, z: lm_true(np.sqrt(r2), np.repeat(0, r2.size), z, cyl=True)
    h_true_uniform = hl.hist_from_eval(f, vectorize=False, bins=1000, range=rang_uniform)
    f = lambda r2, z: lm_again(np.sqrt(r2), np.repeat(0, r2.size), z, cyl=True)
    h_again_uniform = hl.hist_from_eval(f, vectorize=False, bins=1000, range=rang_uniform)
    var = np.var(h_true_uniform.values/h_again_uniform.values)
    mean = np.mean(h_true_uniform.values/h_again_uniform.values)
    acc = np.ndarray.flatten(h_true_uniform.values/h_again_uniform.values)

    # compute lightmap on a grid uniformly in R and Z for plotting
    rang = (rlim[0],rlim[1]), (zlim[0],zlim[1])
    f = lambda r, z: lm_true(r, np.repeat(0, r.size), z, cyl=True)
    h_true = hl.hist_from_eval(f, vectorize=False, bins=1000, range=rang)
    f = lambda r, z: lm_again(r, np.repeat(0, r.size), z, cyl=True)
    h_again = hl.hist_from_eval(f, vectorize=False, bins=1000, range=rang)

    # plot differences between LightMaps
    fig, ax = plt.subplots(figsize=(4,5))
    d = hl.plot2d(ax, h_true / h_again, cbar=True, cmap='RdBu_r',vmin=0.95,vmax=1.05)
    d['colorbar'].set_label(r'True Lightmap / Reconstructed Lightmap')
    ax.set_xlabel(r'$r$ (mm)')
    ax.set_ylabel(r'$z$ (mm)')
    ax.set_title(r'Reconstructed Lightmap Accuracy')
    ax.set_aspect('equal')
    fig.savefig(path+'difference_'+name+'.png',bbox_inches='tight')

    # plot side-by-side of original and retrained LightMap
    fig, axs = plt.subplots(1, 3, figsize=(6,5), gridspec_kw=dict(width_ratios=[1,1,.05]))
    these_lms = lm_true, lm_again
    titles = 'True Lightmap', 'Reconstructed Lightmap',
    for (ax,lm, title) in zip(axs, these_lms, titles):
        d = plot_lm_rz(ax, lm, tpc, cbar=False)
        ax.set_title(title)
    cb = fig.colorbar(d, cax=axs[-1])
    cb.set_label(r'Collection Efficiency')
    for ax in axs[1:-1]:
        ax.set_yticks([])
        ax.set_ylabel('')
    fig.savefig(path+'compare_'+name+'.png',bbox_inches='tight')

    # plot histogram of the reconstructed lightmap accuracy
    fig,ax = plt.subplots(figsize=(4,3))
    ax.hist(acc,bins=100,range=(max([0.8,min(acc)]),min([1.2,max(acc)])),color='navy',histtype=u'step')
    ax.set_title('Reconstructed Lightmap Accuracy')
    ax.set_xlabel('True Lightmap / Reconstructed Lightmap')
    ax.set_ylabel('Counts')
    fig.savefig(path+'accuracy_'+name+'.png',bbox_inches='tight')

    return mean,var,h_true,h_again,h_true_uniform,h_again_uniform
