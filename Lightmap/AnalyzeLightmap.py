import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import pickle
import gzip
# import tqdm
import time

class ReconLightmap:

    def __init__(self,folder):
        """
        Initialize some properties of ReconLightmap object
        """

        # top level folder where all data is saved
        self.folder = folder

        # initialize dataframes for raw and reduced data
        self.raw_data = pd.DataFrame()
        self.reduced_data = pd.DataFrame()


    #--------------- saving and loading functions ---------------#


    def load_datasets(self,prefix,start,end):
        """
        Load all reconstructed lightmaps into a single dataframe
        """
        results_files = []

        # make a list of all results files to loop through
        for i in range(start-1,end):
            results_files.append(self.folder+prefix+str(i+1)+'_results.pkl')

        results = []
        
        # loop through and get all dataframes with reconstructed lightmaps
        for result_file in results_files:
            try:
                result = gzip.open(result_file,'rb')
                this_df = pickle.load(result)
                results.append(this_df)
            except:
                print('Error: failed to load '+result_file)
                print('Skipping and moving to next file')
                pass

        print('Found '+str(len(results))+' datasets')
        results = pd.concat(results,ignore_index=True,sort=False)

        # if raw_data is empty, add the data to it
        if len(self.raw_data.index)==0:
            self.raw_data = results
            print('Raw data loaded and added to dataframe')
            return

        # if the data is already in the dataframe, don't add it again
        duplicates = [x in self.raw_data['name'].values for x in results['name'].values]
        num_dup = sum(duplicates)
        if num_dup > 0:
            print('Ignoring '+str(num_dup)+' dataset(s) already in the dataframe')        
        results = results.loc[list(~np.array(duplicates))]

        # add the data to the existing dataframe
        self.raw_data = pd.concat([self.raw_data,results],ignore_index=True,sort=False)
        print('Raw data loaded and added to dataframe')
                        
        
    def save_dataframe(self,which_df,filename):
        """
        Pickles dataframe. Input 'which_df' is a string identifying which
        dataframe should be pickled
        """
        dfs = [self.raw_data,self.reduced_data]
        try:
            which = dfs[['raw','reduced'].index(which_df)]
        except ValueError:
            print('Error: input "which_df" not recognized')
            return
        pickle.dump(which,open(self.folder+filename,'wb'))
        print('Dataframe saved to '+self.folder+filename)

        
    def load_dataframe(self,which_df,filename):
        """
        Loads dataframe that was previously pickled using 'save_dataframe()'.
        Same inputs as that function
        """
        # loaded = pickle.load(open(self.folder+filename,'rb'))
        loaded = pd.read_pickle(self.folder+filename)
        if which_df == 'raw':
            self.raw_data = loaded
        elif which_df == 'reduced':
            self.reduced_data = loaded
        print('Loaded dataframe from '+self.folder+filename)


    #--------------- utilites to get quantities of interest ---------------#
        
    
    def reduce_raw_data(self,datasets,recompute=False,fiducial_cut=0,parent=None):
        """
        Calculates selected quantities from raw data and saves dataframe
        without raw data or Lightmap reconstruction parameters. Optional
        argument 'parent' specifies the ReconLightmap object storing the
        raw data from which to build the reduced dataframe in case another
        dataframe needs to be created for the same raw data without reloading
        the raw data into memory again
        """
        if parent is None:
            parent = self

        # data to be processed is some subset of raw data
        raw_data = parent.raw_data[parent.raw_data['name'].isin(datasets)].copy()

        # recompute the mean and error and use these new values,
        # otherwise just pass the original values into the new dataframe
        if recompute:
            mean,error = self.compute_from_hist(raw_data,fiducial_cut=fiducial_cut)
            raw_data['mean'] = mean
            raw_data['error'] = error
            
        else:
            raw_data['mean'] = raw_data['accuracy_mean'].values
            raw_data['error'] = raw_data['accuracy_std_dev'].values
        
        reduced_df = pd.concat([raw_data['name'],raw_data['mean'],raw_data['error']],\
                               axis=1,keys=['name','mean','error'])

        # if reduced dataframe is empty, add the new reduced data to it
        if len(self.reduced_data.index)==0:
            self.reduced_data = reduced_df
            print('Data reduced and added to dataframe')
            return

        # if the data is already in the dataframe, don't add it again
        duplicates = [x in self.reduced_data['name'].values for x in reduced_df['name'].values]
        num_dup = sum(duplicates)
        if num_dup > 0:
            print('Ignoring '+str(num_dup)+' dataset(s) already in the dataframe')
        reduced_df = reduced_df.loc[list(~np.array(duplicates))]
        
        self.reduced_data = pd.concat([self.reduced_data,reduced_df],ignore_index=True,sort=False)
        

    def get_param_array(self,datasets,param,which=None):
        """
        Returns an array of a given parameter for the list of datasets
        specified
        """
        
        raw_list = self.raw_data[self.raw_data['name'].isin(datasets)][param].values
            
        if which==None:
            return raw_list
        elif which=='first':
            return raw_list[0]
        elif which=='final_mean':
            return [np.mean([i[-1] for i in j]) for j in raw_list]
        elif which=='length':
            return [len(i) for i in raw_list]
        elif which=='product':
            return [np.prod(i) for i in raw_list]
        elif which=='running_mean':
            return [np.mean(j,axis=0) for j in raw_list][0]
        else:
            print('Error: input variable "which" is not recognized')
            return


    def get_value_array(self,datasets,value):
        """
        Like get_param_array, except for values already calculated
        and put in reduced_data
        """
        return self.reduced_data[self.reduced_data['name'].isin(datasets)][value].values
        

    #--------------- calculate stuff from the raw data ---------------#


    def compute_from_hist(self,df,fiducial_cut=0):
        """
        Recompute the error from the raw lightmap histograms with
        a specific fiducial cut applied
        """
        h_true = df['hist_true_uniform']
        h_again = df['hist_again_uniform']
        standoff = df['fid_cut']

        means = []
        errors = []
        for i in range(len(h_true.index)):
            R_sq,Z = np.meshgrid(h_true.values[i].centers[0],h_true.values[i].centers[1])
            r_max_sq = h_again.values[i].bins[0][-1]
            acc = h_true.values[i].values[R_sq<(np.sqrt(r_max_sq)+standoff.values[i]-fiducial_cut)**2]\
                  /h_again.values[i].values[R_sq<(np.sqrt(r_max_sq)+standoff.values[i]-fiducial_cut)**2]
            means.append(np.mean(acc))
            errors.append(np.std(acc))

        return np.array(means),np.array(errors)
        
    
    #--------------- printing available data ---------------#

    
    def print_columns(self):
        """
        Prints columns of raw dataframe. To be used to determine
        allowed inputs for get_param_array() and other similar
        functions
        """
        print('The dataframe has the following columns:')
        for c in self.raw_data.columns.values:
            print(c)
