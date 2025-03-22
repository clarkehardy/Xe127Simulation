import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import pandas as pd
from os import listdir
from scipy.interpolate import interp1d

##############################################################################
def LoadData():
  data = dict()
  data['Xe124'] = dict()
  data['Xe124']['ng'] = pd.read_csv('CrossSections_copy/Xe124_ng.txt',engine='python',skipfooter=2,header=11,sep='\s+')
  data['Xe125'] = dict()
  data['Xe125']['nnon'] = pd.read_csv('CrossSections_copy/Xe125_nonel.txt',engine='python',skipfooter=2,header=11,sep='\s+')
  data['Xe126'] = dict()
  data['Xe126']['ng'] = pd.read_csv('CrossSections_copy/Xe126_ng.txt',engine='python',skipfooter=2,header=11,sep='\s+')
  data['Xe127']=dict()
  data['Xe127']['nnon'] = pd.read_csv('CrossSections_copy/Xe127_nonel.txt',engine='python',skipfooter=2,header=11,sep='\s+')
  data['Xe136']=dict()
  data['Xe136']['ng'] = pd.read_csv('CrossSections_copy/Xe136_ng.txt',engine='python',skipfooter=2,header=11,sep='\s+')
  data['Xe137']=dict()
  data['Xe137']['nnon'] = pd.read_csv('CrossSections_copy/Xe137_nonel.txt',engine='python',skipfooter=2,header=11,sep='\s+')

  return data

def LoadAllData():
  data = dict()
  for csfile in listdir('CrossSections_copy'):
      filename = csfile.split('.')[0]
      if len(filename.split('_')) < 2: 
         print('Error with {}'.format(filename))
         continue
      isotope = filename.split('_')[0]
      reaction = filename.split('_')[1]
      print('Isotope: {}\tReaction: {}'.format(isotope,reaction))
      if reaction=='ng':
         data[isotope] = dict()
         data[isotope]['ng'] = pd.read_csv('CrossSections_copy/{}'.format(csfile),engine='python',skipfooter=2,header=11,sep='\s+')
      if reaction=='np':
         data[isotope] = dict()
         data[isotope]['np'] = pd.read_csv('CrossSections_copy/{}'.format(csfile),engine='python',skipfooter=2,header=11,sep='\s+')
  return data 
      

def LoadXe129InelasticData():
    data = list()
    for i in range(1,17):
        data.append( pd.read_csv('CrossSections_copy/Xe129_n_inel_{}.txt'.format(i),\
                                 engine='python',skipfooter=2,header=11,sep='\s+') )
    return data

def LoadXe131InelasticData():
    data = list()
    for i in range(1,16):
        data.append( pd.read_csv('CrossSections_copy/Xe131_n_inel_{}.txt'.format(i),\
                                 engine='python',skipfooter=2,header=11,sep='\s+') )
    return data

def LoadSpectrum():
  spectrum = pd.read_csv('Flat_0-11MeV_water_moderated_10x10cm.txt',sep='\s+',engine='python')
  spectrum['Flux'] = spectrum['Counts/1e6']/100.*10. # Divide by 100cm^2, multiply due to 1e7 n/s flux of AmBe. Now in units of n/cm2/s
  return spectrum

def LoadMNRCSpectrum(infile='./mnrc_spectrum.txt'):
  mnrcspectrum = pd.read_csv(infile)
  return mnrcspectrum

#################################################################################
def InterpolateAndIntegrate( csdata, spectrum ):

  integral = 0.
  folded_spectrum = np.zeros(len(spectrum))

  for index, row in spectrum.iterrows():
    energy = row['Energy']
    if index < len(spectrum)-1:
        next_energy = spectrum.loc[index+1,'Energy']
    else:
        next_energy = energy
    if index>0:
        prev_energy = spectrum.loc[index-1,'Energy']
    else:
        prev_energy = energy
    bin_lower_limit = energy - (energy-prev_energy)/2.
    bin_upper_limit = energy + (next_energy-energy)/2.
        
    fluence = row['Flux']
    if fluence == 0.: continue
    below_mask = csdata.iloc[:,0] < energy
    below_idx = len( csdata.iloc[:,0][below_mask])-1
    #print('Energy: {}, below_idx: {}'.format(energy,below_idx))i
    cs = (csdata.iloc[:,1].iloc[below_idx] + csdata.iloc[:,1].iloc[below_idx+1])/2.
    folded_spectrum[index] = cs * 1.e-24 * fluence
    integral = integral + cs * 1.e-24 * fluence

  return integral, folded_spectrum

#############################################################################
def InterpolateAndIntegrateAvg( csdata, spectrum, units='b' ):

  integral = 0.
  folded_spectrum = np.zeros(len(spectrum))

  for index, row in spectrum.iterrows():
    energy = row['Energy']
    if index < len(spectrum)-1:
        next_energy = spectrum.loc[index+1,'Energy']
    else:
        next_energy = energy
    if index>0:
        prev_energy = spectrum.loc[index-1,'Energy']
    else:
        prev_energy = energy
    bin_lower_limit = energy - (energy-prev_energy)/2.
    bin_upper_limit = energy + (next_energy-energy)/2.
        
    fluence = row['Flux']
    if fluence == 0.: continue
    below_mask = csdata.iloc[:,0] < bin_lower_limit
    above_mask = csdata.iloc[:,0] < bin_upper_limit
    below_idx = len( csdata.iloc[:,0][below_mask])-1
    above_idx = len( csdata.iloc[:,0][above_mask])-1
    #print('Energy: {}, below_idx: {}'.format(energy,below_idx))
    if units=='mb':
       cs_point = (csdata.iloc[:,1].iloc[below_idx] + csdata.iloc[:,1].iloc[below_idx+1])/2./1000.
       cs = np.mean( csdata.iloc[below_idx:above_idx,1] )/1000.
    elif units=='b':
       cs_point = (csdata.iloc[:,1].iloc[below_idx] + csdata.iloc[:,1].iloc[below_idx+1])/2.
       cs = np.mean( csdata.iloc[below_idx:above_idx,1] )
    else:
      raise ValueError('{} not a valid unit! Please choose b or mb.'.format(units))
    folded_spectrum[index] = cs * 1.e-24 * fluence
    if folded_spectrum[index] != folded_spectrum[index]:
        integral = integral + cs_point * 1.e-24 * fluence
    else:
        integral = integral + cs * 1.e-24 * fluence

  return integral, folded_spectrum


###############################################################################
def InterpolateAndIntegrateAvg2( csdata, spectrum, units='b' ):

  csfunc = interp1d( np.log10(csdata.iloc[:,0]), np.log10(csdata.iloc[:,1]) )
    
  integral = 0.
  folded_spectrum = np.zeros(len(spectrum))

  for index, row in spectrum.iterrows():
    energy = row['Energy']
    if index < len(spectrum)-1:
        next_energy = spectrum.loc[index+1,'Energy']
    else:
        next_energy = energy
    if index>0:
        prev_energy = spectrum.loc[index-1,'Energy']
    else:
        prev_energy = energy
    bin_lower_limit = energy - (energy-prev_energy)/2.
    bin_upper_limit = energy + (next_energy-energy)/2.
        
    fluence = row['Flux']
    if fluence == 0.: continue
        
    npts=1000
    xvals = np.linspace(bin_lower_limit,bin_upper_limit,npts)
    
    cs_avg = np.sum(10**csfunc(np.log10(xvals)))/npts
        
    below_mask = csdata.iloc[:,0] < bin_lower_limit
    above_mask = csdata.iloc[:,0] < bin_upper_limit
    below_idx = len( csdata.iloc[:,0][below_mask])-1
    above_idx = len( csdata.iloc[:,0][above_mask])-1
    #print('Energy: {}, below_idx: {}'.format(energy,below_idx))
    cs_point = (csdata.iloc[:,1].iloc[below_idx] + csdata.iloc[:,1].iloc[below_idx+1])/2.

    if units=='mb':
       cs_point = cs_point / 1000.
       cs_avg = cs_avg / 1000.
    elif units=='b':
       pass
    else:
      raise ValueError('{} not a valid unit! Please choose b or mb.'.format(units))

    #cs = np.mean( csdata.iloc[below_idx:above_idx,1] )
    folded_spectrum[index] = cs_avg * 1.e-24 * fluence
    if folded_spectrum[index] != folded_spectrum[index]:
        integral = integral + cs_point * 1.e-24 * fluence
    else:
        integral = integral + cs_avg * 1.e-24 * fluence

  return integral, folded_spectrum


###############################################################################      
def InterpolateNucData( csdata, energy_array ):
    interpolated_data = np.zeros(len(energy_array))
    for i in range(0,len(energy_array)):
        below_mask = csdata.iloc[:,0] < energy_array[i]
        below_idx = len(csdata.iloc[:,0][below_mask])-1
        if below_idx < 0.: continue
        if below_idx+1 == len(csdata.iloc[:,0]): continue
        interpolated_data[i] = csdata.iloc[:,1].iloc[below_idx] +\
                               (csdata.iloc[:,1].iloc[below_idx+1]-csdata.iloc[:,1].iloc[below_idx]) / \
                               (csdata.iloc[:,0].iloc[below_idx+1]-csdata.iloc[:,0].iloc[below_idx]) * \
                               (energy_array[i] - csdata.iloc[:,0].iloc[below_idx])
        
    return interpolated_data 
            


def RunAmBeXe127():
   nucdata = LoadData()
   ambespec = LoadSpectrum()
   
   # Assume 1kg of xenon.
   N126_0 = 1000./131. * 0.00089 * 6.02e23
   N127_0 = 0.
   tau127 = 36.3 / np.log(2.) * 24. * 60 * 60.
   
   time = np.linspace(0.,2500000.,1000000)
   dt = time[2]-time[1]
   
   N127 = np.zeros(len(time))
   
   creation_int = InterpolateAndIntegrate( nucdata['Xe126']['ng'], ambespec )
   destruction_int = InterpolateAndIntegrate( nucdata['Xe127']['nnon'], ambespec )

   #print(creation_int)
   #print(destruction_int)
   #print(dt)  
 
   for i in range(1,len(time)):
     term1 = -N127[i-1]/tau127
     term2 = N126_0 * creation_int
     term3 = -N127[i-1] * destruction_int
     N127[i] = N127[i-1] + dt * (term1 + term2 + term3)
    # print(N127[i])

   plt.figure(1)
   plt.plot(time/60./60./24.,N127/tau127,'-b')
   plt.xlabel('Irradiation time (days)')
   plt.ylabel('Xe127 Activity (Bq/kg)')
   plt.title('')
   plt.xscale('linear')
   plt.yscale('log')
   plt.axis([0.,30.,1.e-1,1.e2])   
   plt.savefig('/Users/blenardo/Research/nEXO/ActivatedXenonCalibrations/ActivationCalculation/Xe127_AmBe_activation_vs_time.png',dpi=300) 
 
   return time, N127


def RunMNRCXe127():
   nucdata = LoadData()
   ambespec = LoadSpectrum()
   ambespec['Flux'] = ambespec['Flux']*10000.

   # Assume 1kg of xenon.
   N126_0 = 1000./131. * 0.00089 * 6.02e23
   N127_0 = 0.
   tau127 = 36.3 / np.log(2.) * 24. * 60 * 60.
   
   time = np.linspace(0.,2500000.,1000000)
   dt = time[2]-time[1]
   
   N127 = np.zeros(len(time))
   
   creation_int = InterpolateAndIntegrate( nucdata['Xe126']['ng'], ambespec )
   destruction_int = InterpolateAndIntegrate( nucdata['Xe127']['nnon'], ambespec )

   #print(creation_int)
   #print(destruction_int)
   #print(dt)  
 
   for i in range(1,len(time)):
     term1 = -N127[i-1]/tau127
     term2 = N126_0 * creation_int
     term3 = -N127[i-1] * destruction_int
     N127[i] = N127[i-1] + dt * (term1 + term2 + term3)
    # print(N127[i])
   
   
#   plt.plot(time/60./60./24.,N126,'-b')
#   plt.xlabel('Time (days)')
#   plt.ylabel('Xe127 Activity (Bq/kg)')
#   plt.title('')
#   plt.xscale('linear')
#   plt.yscale('log')

   return time, N127

def PlotBothXe127():

   tau127 = 36.3 / np.log(2.) * 24. * 60 * 60.
  
   time, N127_AmBe = RunAmBeXe127()
   time, N127_MNRC = RunMNRCXe127() 

   plt.figure(2)
   plt.plot(time/60./60./24.,N127_AmBe/tau127,label='Stanford AmBe source')
   plt.plot(time/60./60./24.,N127_MNRC/tau127,label='MNRC Reactor')
   plt.xlabel('Irradiation time (days)')
   plt.ylabel('Xe127 Activity (Bq/kg)')
   plt.title('')
   plt.legend(loc='lower right')
   plt.xscale('linear')
   plt.yscale('log')
