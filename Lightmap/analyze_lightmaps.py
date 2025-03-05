import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import pickle
import gzip
import argparse

# set plotting style
plt.rc('figure', dpi=200, figsize=(6,4), facecolor='w')
plt.rc('savefig', dpi=200, facecolor='w')
plt.rc('lines', linewidth=1.5)

'''
Arguments:
-input_files   :    List of processed simulation files
-fid_cut       :    True if running on data with fiducial cut applied
'''
parser = argparse.ArgumentParser()
parser.add_argument('-input_files',type=str,nargs='+')
parser.add_argument('-fid_cut',action='store_true',default=False)
args = parser.parse_args()
results_files = args.input_files
fid_cut = args.fid_cut

results = []

# loop through and get all results files
for result_file in results_files:
    result = gzip.open(result_file,'rb')
    this_df = pickle.load(result)
    results.append(this_df)

results = pd.concat(results,ignore_index=True)
print('\nResults files processed.\n')

# extract error, loss, and layers
if fid_cut:
    plot_1 = ['nn_57','nn_58','nn_59']
    plot_2 = ['nn_60','nn_61','nn_62']
    plot_3 = ['nn_63','nn_64','nn_65']
    plot_4 = ['nn_66','nn_67','nn_68']
else:
    plot_1 = ['nn_1','nn_2','nn_3']
    plot_2 = ['nn_4','nn_5','nn_6']
    plot_3 = ['nn_7','nn_8','nn_9']
    plot_4 = ['nn_10','nn_11','nn_12']
std_1 = results[results['name'].isin(plot_1)]['accuracy_std_dev'].values
std_2 = results[results['name'].isin(plot_2)]['accuracy_std_dev'].values
std_3 = results[results['name'].isin(plot_3)]['accuracy_std_dev'].values
std_4 = results[results['name'].isin(plot_4)]['accuracy_std_dev'].values
lr_1 =  results[results['name'].isin(plot_1)]['learning_rate'].values[0]
lr_2 =	results[results['name'].isin(plot_2)]['learning_rate'].values[0]
lr_3 =	results[results['name'].isin(plot_3)]['learning_rate'].values[0]
lr_4 =	results[results['name'].isin(plot_4)]['learning_rate'].values[0]
num_layers_1 = [len(i) for i in results[results['name'].isin(plot_1)]['layers'].values]
num_layers_2 = [len(i) for i in results[results['name'].isin(plot_2)]['layers'].values]
num_layers_3 = [len(i) for i in results[results['name'].isin(plot_3)]['layers'].values]
num_layers_4 = [len(i) for i in results[results['name'].isin(plot_4)]['layers'].values]
loss_1 = [np.mean([i[-1] for i in j]) for j in results[results['name'].isin(plot_1)]['losses'].values]
loss_2 = [np.mean([i[-1] for i in j]) for j in results[results['name'].isin(plot_2)]['losses'].values]
loss_3 = [np.mean([i[-1] for i in j]) for j in results[results['name'].isin(plot_3)]['losses'].values]
loss_4 = [np.mean([i[-1] for i in j]) for j in results[results['name'].isin(plot_4)]['losses'].values]

# plot error and loss vs number of layers for different learning rates
fig,ax = plt.subplots()
#ax.plot(num_layers_1,std_1,label=str(lr_1))
ln1_2 = ax.plot(num_layers_2,100*std_2,label='Error, lr={:.0e}'.format(lr_2),color='lightcoral')
ln1_3 = ax.plot(num_layers_3,100*std_3,label='Error, lr={:.0e}'.format(lr_3),color='crimson')
#ax.plot(num_layers_4,std_4,label=str(lr_4))
ax.set_xlabel('Number of layers')
ax.set_ylabel('Lightmap error (%)')
ax.set_xticks([3,4,5])
ax2 = ax.twinx()
ax2.set_ylabel('Loss')
#ax2.plot(num_layers_1,loss_1,label=str(lr_1))
ln2_2 = ax2.plot(num_layers_2,loss_2,label='Loss, lr={:.0e}'.format(lr_2),color='skyblue',ls='--')
ln2_3 = ax2.plot(num_layers_3,loss_3,label='Loss, lr={:.0e}'.format(lr_3),color='mediumblue',ls='--')
#ax2.plot(num_layers_4,loss_4,label=str(lr_4))
lns = ln1_2+ln1_3+ln2_2+ln2_3
labs = [l.get_label() for l in lns]
ax2.legend(lns,labs,loc='best')
ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
plt.tight_layout()

# extract error, loss, and number of nodes
if fid_cut:
    plot_5 = ['nn_69','nn_70','nn_71','nn_72']
    plot_6 = ['nn_73','nn_74','nn_75','nn_76']
    plot_7 = ['nn_77','nn_78','nn_79','nn_80']
    plot_8 = ['nn_81','nn_82','nn_83','nn_84']
else:
    plot_5 = ['nn_13','nn_14','nn_15','nn_16']
    plot_6 = ['nn_17','nn_18','nn_19','nn_20']
    plot_7 = ['nn_21','nn_22','nn_23','nn_24']
    plot_8 = ['nn_25','nn_26','nn_27','nn_28']
std_5 = results[results['name'].isin(plot_5)]['accuracy_std_dev'].values
std_6 = results[results['name'].isin(plot_6)]['accuracy_std_dev'].values
std_7 = results[results['name'].isin(plot_7)]['accuracy_std_dev'].values
std_8 = results[results['name'].isin(plot_8)]['accuracy_std_dev'].values
lr_5 =  results[results['name'].isin(plot_5)]['learning_rate'].values[0]
lr_6 =  results[results['name'].isin(plot_6)]['learning_rate'].values[0]
lr_7 =  results[results['name'].isin(plot_7)]['learning_rate'].values[0]
lr_8 =  results[results['name'].isin(plot_8)]['learning_rate'].values[0]
num_nodes_5 = [np.prod(i) for i in results[results['name'].isin(plot_5)]['layers'].values]
num_nodes_6 = [np.prod(i) for i in results[results['name'].isin(plot_6)]['layers'].values]
num_nodes_7 = [np.prod(i) for i in results[results['name'].isin(plot_7)]['layers'].values]
num_nodes_8 = [np.prod(i) for i in results[results['name'].isin(plot_8)]['layers'].values]
loss_5 = [np.mean([i[-1] for i in j]) for j in results[results['name'].isin(plot_5)]['losses'].values]
loss_6 = [np.mean([i[-1] for i in j]) for j in results[results['name'].isin(plot_6)]['losses'].values]
loss_7 = [np.mean([i[-1] for i in j]) for j in results[results['name'].isin(plot_7)]['losses'].values]
loss_8 = [np.mean([i[-1] for i in j]) for j in results[results['name'].isin(plot_8)]['losses'].values]

# plot error and loss vs number of layers for different learning rates
fig,ax = plt.subplots()
#ax.plot(num_layers_1,std_1,label=str(lr_1))
ln1_6 = ax.semilogx(num_nodes_6,100*std_6,label='Error, lr={:.0e}'.format(lr_6),color='lightcoral')
ln1_7 = ax.semilogx(num_nodes_7,100*std_7,label='Error, lr={:.0e}'.format(lr_7),color='crimson')
#ax.plot(num_layers_4,std_4,label=str(lr_4))
ax.set_xlabel('Number of nodes')
ax.set_ylabel('Lightmap error (%)')
ax2 = ax.twinx()
ax2.set_ylabel('Loss')
#ax2.plot(num_layers_1,loss_1,label=str(lr_1))
ln2_6 = ax2.semilogx(num_nodes_6,loss_6,label='Loss, lr={:.0e}'.format(lr_6),color='skyblue',ls='--')
ln2_7 = ax2.semilogx(num_nodes_7,loss_7,label='Loss, lr={:.0e}'.format(lr_7),color='mediumblue',ls='--')
#ax2.plot(num_layers_4,loss_4,label=str(lr_4))
lns = ln1_6+ln1_7+ln2_6+ln2_7
labs = [l.get_label() for l in lns]
ax2.legend(lns,labs,loc='best')
ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
plt.tight_layout()

# extract error, loss, and batch size
if fid_cut:
    plot_9 = ['nn_85','nn_86','nn_87','nn_88','nn_89','nn_90']
    plot_10 = ['nn_91','nn_92','nn_93','nn_94','nn_95','nn_96']
    plot_11 = ['nn_97','nn_98','nn_99','nn_100','nn_101','nn_102']
    plot_12 = ['nn_103','nn_104','nn_105','nn_106','nn_107','nn_108']
else:
    plot_9 = ['nn_29','nn_30','nn_31','nn_32','nn_33','nn_34']
    plot_10 = ['nn_35','nn_36','nn_37','nn_38','nn_39','nn_40']
    plot_11 = ['nn_41','nn_42','nn_43','nn_44','nn_45','nn_46']
    plot_12 = ['nn_47','nn_48','nn_49','nn_50','nn_51','nn_52']
std_9 = results[results['name'].isin(plot_9)]['accuracy_std_dev'].values
std_10 = results[results['name'].isin(plot_10)]['accuracy_std_dev'].values
std_11 = results[results['name'].isin(plot_11)]['accuracy_std_dev']
std_12 = results[results['name'].isin(plot_12)]['accuracy_std_dev'].values
lr_9 = results[results['name'].isin(plot_9)]['learning_rate'].values[0]
lr_10 = results[results['name'].isin(plot_10)]['learning_rate'].values[0]
lr_11 = results[results['name'].isin(plot_11)]['learning_rate'].values[0]
lr_12 = results[results['name'].isin(plot_12)]['learning_rate'].values[0]
bs_9 = results[results['name'].isin(plot_9)]['batch_size'].values
bs_10 = results[results['name'].isin(plot_10)]['batch_size'].values
bs_11 = results[results['name'].isin(plot_11)]['batch_size'].values
bs_12 = results[results['name'].isin(plot_12)]['batch_size'].values
loss_9 = [np.mean([i[-1] for i in j]) for j in results[results['name'].isin(plot_9)]['losses'].values]
loss_10 = [np.mean([i[-1] for i in j]) for j in results[results['name'].isin(plot_10)]['losses'].values]
loss_11 = [np.mean([i[-1] for i in j]) for j in results[results['name'].isin(plot_11)]['losses'].values]
loss_12 = [np.mean([i[-1] for i in j]) for j in results[results['name'].isin(plot_12)]['losses'].values]

std_11 = np.array([x for _, x in sorted(zip(bs_11,std_11))])
loss_11 = np.array([x for _, x in sorted(zip(bs_11,loss_11))])
bs_11 = sorted(bs_11)

# plot error and loss vs batch size for different learning rates
fig,ax = plt.subplots()
#ax.plot(num_layers_1,std_1,label=str(lr_1))
ln1_10 = ax.semilogx(bs_10,100*std_10,label='Error, lr={:.0e}'.format(lr_10),color='lightcoral')
ln1_11 = ax.semilogx(bs_11,100*std_11,label='Error, lr={:.0e}'.format(lr_11),color='crimson')
#ax.plot(num_layers_4,std_4,label=str(lr_4))
ax.set_xlabel('Batch size')
ax.set_ylabel('Lightmap error (%)')
ax2 = ax.twinx()
ax2.set_ylabel('Loss')
#ax2.plot(num_layers_1,loss_1,label=str(lr_1))
ln2_10 = ax2.semilogx(bs_10,loss_10,label='Loss, lr={:.0e}'.format(lr_10),color='skyblue',ls='--')
ln2_11 = ax2.semilogx(bs_11,loss_11,label='Loss, lr={:.0e}'.format(lr_11),color='mediumblue',ls='--')
#ax2.plot(num_layers_4,loss_4,label=str(lr_4))
lns = ln1_10+ln1_11+ln2_10+ln2_11
labs = [l.get_label() for l in lns]
ax2.legend(lns,labs,loc='best')
ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
plt.tight_layout()

# extract error, loss, and size of ensemble
if fid_cut:
    plot_13 = ['nn_109','nn_110','nn_111','nn_112']
else:
    plot_13 = ['nn_53','nn_54','nn_55','nn_56']
std_13 = results[results['name'].isin(plot_13)]['accuracy_std_dev'].values
es_13 = results[results['name'].isin(plot_13)]['ensemble_size'].values
loss_13 = [np.mean([i[-1] for i in j]) for j in results[results['name'].isin(plot_13)]['losses'].values]

# plot error and loss vs size of ensemble
fig,ax = plt.subplots()
ln1_13 = ax.plot(es_13,100*std_13,label='Error',color='lightcoral')
ax.set_xlabel('Ensemble size')
ax.set_ylabel('Lightmap error (%)')
ax2 = ax.twinx()
ax2.set_ylabel('Loss')
ln2_13 = ax2.plot(es_13,loss_13,label='Loss',color='skyblue',ls='--')
lns = ln1_13+ln2_13
labs = [l.get_label() for l in lns]
ax2.legend(lns,labs,loc='best')
ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
plt.tight_layout()

# extract error, loss, and number of events
if fid_cut:
    plot_14 = ['nn_121','nn_122','nn_123','nn_124']
    plot_15 = ['nn_125','nn_126','nn_127','nn_128']
else:
    plot_14 = ['nn_113','nn_114','nn_115','nn_116']
    plot_15 = ['nn_117','nn_118','nn_119','nn_120']
std_14 = results[results['name'].isin(plot_14)]['accuracy_std_dev'].values
std_15 = results[results['name'].isin(plot_15)]['accuracy_std_dev'].values
evts_14 = results[results['name'].isin(plot_14)]['num_events'].values
evts_15 = results[results['name'].isin(plot_15)]['num_events'].values
peaks_14 = results[results['name'].isin(plot_14)]['num_peaks'].values[0]
peaks_15 = results[results['name'].isin(plot_15)]['num_peaks'].values[0]
loss_14 = [np.mean([i[-1] for i in j]) for j in results[results['name'].isin(plot_14)]['losses'].values]
loss_15 = [np.mean([i[-1] for i in j]) for j in results[results['name'].isin(plot_15)]['losses'].values]

# plot error and loss vs number of events
fig,ax = plt.subplots()
ln1_14 = ax.semilogx(evts_14,100*std_14,label='Error, {:d} peak'.format(peaks_14),color='lightcoral')
ln1_15 = ax.semilogx(evts_15,100*std_15,label='Error, {:d} peak'.format(peaks_15),color='crimson')
ax.set_xlabel('Number of events')
ax.set_ylabel('Lightmap error (%)')
ax2 = ax.twinx()
ax2.set_ylabel('Loss')
ln2_14 = ax2.semilogx(evts_14,loss_14,label='Loss, {:d} peak'.format(peaks_14),color='skyblue',ls='--')
ln2_15 = ax2.semilogx(evts_15,loss_15,label='Loss, {:d} peak'.format(peaks_15),color='mediumblue',ls='--')
lns = ln1_14+ln1_15+ln2_14+ln2_15
labs = [l.get_label() for l in lns]
ax2.legend(lns,labs,loc='best')
ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
plt.tight_layout()

# extract error, loss, and smoothing length
if fid_cut:
    plot_16 = ['ks_9','ks_10','ks_11','ks_12','ks_13','ks_14','ks_15','ks_16']
else:
    plot_16 = ['ks_1','ks_2','ks_3','ks_4','ks_5','ks_6','ks_7','ks_8']
std_16 = results[results['name'].isin(plot_16)]['accuracy_std_dev'].values
sigma_16 = results[results['name'].isin(plot_16)]['sigma'].values
std_16 = np.array([x for _, x in sorted(zip(sigma_16,std_16))])
sigma_16 = sorted(sigma_16)

# plot error and loss vs smoothing length
fig,ax = plt.subplots()
ln1_16 = ax.plot(sigma_16,100*std_16,label='Error',color='lightcoral')
ax.set_xlabel('Smoothing length (mm)')
ax.set_ylabel('Lightmap error (%)')
plt.tight_layout()

# extract error, loss, and number of events
if fid_cut:
    plot_17 = ['ks_17','ks_18','ks_19','ks_20']
    plot_18 = ['ks_21','ks_22','ks_23','ks_24']
else:
    plot_17 = ['ks_25','ks_26','ks_27','ks_28']
    plot_18 = ['ks_29','ks_30','ks_31','ks_32']
std_17 = results[results['name'].isin(plot_14)]['accuracy_std_dev'].values
std_18 = results[results['name'].isin(plot_15)]['accuracy_std_dev'].values
evts_17 = results[results['name'].isin(plot_14)]['num_events'].values
evts_18 = results[results['name'].isin(plot_15)]['num_events'].values
peaks_17 = results[results['name'].isin(plot_14)]['num_peaks'].values[0]
peaks_18 = results[results['name'].isin(plot_15)]['num_peaks'].values[0]

# plot error and loss vs number of events
fig,ax = plt.subplots()
ln1_17 = ax.semilogx(evts_17,100*std_17,label='Error, {:d} peak'.format(peaks_17),color='lightcoral')
ln1_18 = ax.semilogx(evts_18,100*std_18,label='Error, {:d} peak'.format(peaks_18),color='crimson')
ax.set_xlabel('Number of events')
ax.set_ylabel('Lightmap error (%)')
ax.legend(loc='best')
plt.tight_layout()


fig,ax = plt.subplots()
errs = []
lrs = []
for i in range(22):
    plot_19 = ['nn_{:.0f}'.format(129+i)]
    std_19 = results[results['name'].isin(plot_19)]['accuracy_std_dev'].values
    errs.append(std_19)
    lr_19 =  results[results['name'].isin(plot_19)]['learning_rate'].values[0]
    lrs.append(lr_19)
    loss_19 = [np.mean(j,axis=0) for j in results[results['name'].isin(plot_19)]['losses'].values][0]
    ax.semilogy(loss_19,label=str(lr_19))

errs = np.array([x for _, x in sorted(zip(lrs,errs))])
lrs = sorted(lrs)

ax.legend(loc='best',prop={'size': 6})
ax.set_xlabel('Epoch')
ax.set_ylabel('Training loss')

fig,ax = plt.subplots()
ax.semilogx(lrs,errs)
ax.set_xlabel('Learning rate')
ax.set_ylabel('Lightmap error')

errs = []
bss = []
lss = []
for i in range(8):
    plot_20 = ['nn_{:.0f}'.format(151+i)]
    std_20 = results[results['name'].isin(plot_20)]['accuracy_std_dev'].values
    errs.append(std_20)
    bs_20 =  results[results['name'].isin(plot_20)]['batch_size'].values[0]
    bss.append(bs_20)
    loss_20 = [np.mean(j,axis=0) for j in results[results['name'].isin(plot_20)]['losses'].values][0][-1]
    lss.append(loss_20)

bss = np.array(bss)
lss = np.array(lss)
errs = np.array(errs)
    
# plot error and loss vs number of events
fig,ax = plt.subplots()
ln1_20 = ax.semilogx(bss,100*errs,label='Error',color='lightcoral')
ax.set_xlabel('Batch size')
ax.set_ylabel('Lightmap error (%)')
ax2 = ax.twinx()
ax2.set_ylabel('Loss')
ln2_20 = ax2.semilogx(bss,lss,label='Loss',color='skyblue',ls='--')
lns = ln1_20+ln2_20
labs = [l.get_label() for l in lns]
ax2.legend(lns,labs,loc='best')
ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
plt.tight_layout()

plt.show()
