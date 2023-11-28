import os
import matplotlib.pyplot as plt
import re

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
import time

from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.ticker                         # here's where the formatter is
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from astropy.io import fits
from astropy.time import Time

from scipy.optimize import curve_fit
from scipy.ndimage import median_filter,uniform_filter,gaussian_filter
from scipy.signal import medfilt2d


# LSST Display
import lsst.afw.display as afwDisplay
afwDisplay.setDefaultBackend('matplotlib')


# Butler
import lsst.daf.butler as dafButler
from lsst.daf.butler import CollectionType
#from lsst.daf.butler import CollectionTypes



def find_flat_dates(obs_year=None,cameraName='LATISS',filter='empty',disperser='empty',obs_type='flat',repo_embargo=True,
                    calibCollections=['LATISS/calib','LATISS/raw/all']):
    
    physical_filter = '{0}~{1}'.format(filter,disperser)
    
    if repo_embargo:
        repo = "/sdf/group/rubin/repo/oga/"
    else:
        repo = "/sdf/group/rubin/repo/main/"
    
    butler = dafButler.Butler(repo)
    registry = butler.registry

    query = "instrument='{0}' AND physical_filter='{1}' AND exposure.observation_type='{2}'".format(cameraName,physical_filter,obs_type)
    flat_records = list(registry.queryDimensionRecords('exposure',where=query))
    
    if len(flat_records)==0:
        print('WARNING: No flats found with the selected settings: \n Camera = {0} \n Physical filter = {1} \n Observation type = {2}'.format(cameraName,physical_filter,obs_type))
        flat_dates = None
        flat_ids = None
    else:
        flat_dates = []
        ids_ = []
        for flat_ in flat_records:
            if obs_year is not None:
                if str(obs_year) in str(flat_.day_obs):
                    flat_dates.append(flat_.day_obs)
                    ids_.append(flat_.id)
            else:
                flat_dates.append(flat_.day_obs)
                ids_.append(flat_.id)

        flat_dates = np.sort(np.unique(flat_dates))

        flat_ids = {}
        for date_ in flat_dates:
            flat_ids[date_] = []
            for id_ in ids_:
                if str(date_) in str(id_):
                    flat_ids[date_].append(id_)
            flat_ids[date_] = np.sort(flat_ids[date_])
        
    return flat_dates, flat_ids, butler



def find_closest_date(date,flat_ids=None,repo_embargo=True,cameraName='LATISS',filter='empty',disperser='empty',
                        calibCollections=['LATISS/calib','LATISS/raw/all'],obs_type='flat'):

    obs_day = int(date)
    
    print('Requested observation date = ', obs_day)
    
    if flat_ids is not None:
        assert isinstance(flat_ids,dict)
        flat_dates = np.array(list(flat_ids.keys()))
    
    else:
        obs_year = date[:4]
        flat_dates, flat_ids, _ = find_flat_dates(obs_year=obs_year,cameraName=cameraName,filter=filter,disperser=disperser,
                                                        obs_type=obs_type,repo_embargo=repo_embargo,calibCollections=calibCollections)
    
    date_diff = int(obs_day)-flat_dates
    date_diff = date_diff[date_diff>=0]
    closest_idx = np.where(date_diff==np.min(date_diff))[0][0]
    closest_date = flat_dates[closest_idx]
    closest_ids = flat_ids[closest_date]
    print('Closest date available = ',closest_date)
    print('Corresponding flat IDs = ', closest_ids)
    
    return closest_date, closest_ids


def check_flats(flat_ids,return_flats=True,butler=None,repo_embargo=True,cameraName='LATISS',detector=0,calibCollections=['LATISS/calib','LATISS/raw/all'],obs_type='flat'):
    
    flat_ids = list(flat_ids)

    if butler is None:
        print('ATTENTION: butler was not set up. Setting it up now.')
        if repo_embargo:
            repo = "/sdf/group/rubin/repo/oga/"
        else:
            repo = "/sdf/group/rubin/repo/main/"
        butler = dafButler.Butler(repo)
    
    flat_array_dict = {}
    if len(flat_ids)==1:
        print('Only one flat ID given. Only checking if it can be loaded')
        try:
            flat_img_ = butler.get(obs_type,instrument=cameraName, exposure=flat_ids[0], detector=detector, collections=calibCollections)
            print('Flat {0} properly loaded'.format(flat_ids[0]))
            flat_array_dict[flat_ids[0]] = flat_img_.getImage().array
        except:
            print('Could not load flat {0}'.format(flat_ids[0]))
        
    else:
        print('Inspecting flats')
        flat_array_list = []
        for iflat_ in flat_ids:
            flat_img_ = butler.get(obs_type,instrument=cameraName, exposure=iflat_, detector=detector, collections=calibCollections)
            flat_array_list.append(flat_img_.getImage().array)
            flat_array_dict[iflat_] = flat_img_.getImage().array
            del(flat_img_)
        
        all_equal = []
        for i,iflat_ in enumerate(flat_array_list):
            for j,jflat_ in enumerate(flat_array_list):
                if j!=i:
                    equal_ = (jflat_.flatten()==iflat_.flatten()).all()
                    all_equal.append(equal_)
                
        all_equal = np.array(all_equal)
        if (all_equal==True).all():
            print('All flats are equal')
        else:
            print('ATTENTION: not all flats are equal. THIS NEEDS TO BE CODED')
        
    if return_flats:
        return flat_array_dict
    else:
        return


def get_amplis_coords(flat_img):
    
    amplis_coords = {}

    for ampIdx, amp in enumerate(flat_img.getDetector()):
        ampName = amp.getName()
        xbegin = amp.getBBox().x.begin
        xend = amp.getBBox().x.end
        ybegin = amp.getBBox().y.begin
        yend = amp.getBBox().y.end
        amplis_coords[ampName] = (xbegin,xend,ybegin,yend)
        
    return amplis_coords


def get_flat_metadata(flat_id=None,flat_img=None,butler=None):
    
    if flat_id is None and flat_img is None:
        print('ATTENTION: Both flat_id and flat_img are None. Doing nothing.')
        return

    if flat_id is not None and flat_img is not None:
        print('ATTENTION: Both flat_id and flat_img are not None.')
        print('Setting flat_id = None and using flat_img information')
        flat_id = None

    if flat_id is not None:
        if butler is None:
            print('ATTENTION: butler was not set up. Setting it up now.')
            if repo_embargo:
                repo = "/sdf/group/rubin/repo/oga/"
            else:
                repo = "/sdf/group/rubin/repo/main/"
            butler = dafButler.Butler(repo)
        
        try:
            flat_img_ = butler.get(obs_type,instrument=cameraName, exposure=flat_id, detector=detector, collections=calibCollections)
            print('Flat {0} properly loaded'.format(flat_id))
            metadata = flat_img_.getMetadata().toDict()
            del(flat_img_)
            
            return metadata
        except:
            print('Could not load flat {0}'.format(flat_id))
            return
       
    else:
        metadata = flat_img.getMetadata().toDict()

        return metadata



def get_flat_array(flat_id=None,flat_img=None,butler=None,repo_embargo=True,cameraName='LATISS',detector=0,calibCollections=['LATISS/calib','LATISS/raw/all'],obs_type='flat'):
    
    if flat_id is None and flat_img is None:
        print('ATTENTION: Both flat_id and flat_img are None. Doing nothing.')
        return

    if flat_id is not None and flat_img is not None:
        print('ATTENTION: Both flat_id and flat_img are not None.')
        print('Setting flat_id = None and using flat_img information')
        flat_id = None
        
    if flat_id is not None:
        if butler is None:
            print('ATTENTION: butler was not set up. Setting it up now.')
            if repo_embargo:
                repo = "/sdf/group/rubin/repo/oga/"
            else:
                repo = "/sdf/group/rubin/repo/main/"
            butler = dafButler.Butler(repo)
        
        try:
            flat_img_ = butler.get(obs_type,instrument=cameraName, exposure=flat_id, detector=detector, collections=calibCollections)
            print('Flat {0} properly loaded'.format(flat_id))
            flat_array = flat_img_.image.array
            del(flat_img_)

            return flat_array
        except:
            print('Could not load flat {0}'.format(flat_id))
            return

    else:
        flat_array = flat_img.image.array

        return flat_array 


def cut_flat_array(flat_img,amplis):

    if amplis=='all':
        print('All amplifiers are selected. Nothing to cut')
        return 
    else:
        flat_array_ = flat_img.image.array
        cut_array_ = np.ones(flat_array_.shape)
        amplis_coords_ = get_amplis_coords(flat_img)
        
        if isinstance(amplis,str):
            sel_amplis = [amplis]
        else:
            sel_amplis = list(amplis)
        for ampli_ in sel_amplis:
            x0_ = amplis_coords_[ampli_][0]
            x1_ = amplis_coords_[ampli_][1]
            y0_ = amplis_coords_[ampli_][2]
            y1_ = amplis_coords_[ampli_][3]
            
            cut_array_[y0_:y1_,x0_:x1_] = flat_array_[y0_:y1_,x0_:x1_]
            
        return cut_array_


def normalize_flat_array(flat_img,amplis):
    
    flat_array_ = flat_img.image.array
    norm_array_ = np.ones(flat_array_.shape)
    amplis_coords_ = get_amplis_coords(flat_img)
    
    if isinstance(amplis,str):
        sel_amplis = [amplis]
    else:
        sel_amplis = list(amplis)
    for ampli_ in sel_amplis:
        x0_ = amplis_coords_[ampli_][0]
        x1_ = amplis_coords_[ampli_][1]
        y0_ = amplis_coords_[ampli_][2]
        y1_ = amplis_coords_[ampli_][3]
        
        norm_array_[y0_:y1_,x0_:x1_] = flat_array_[y0_:y1_,x0_:x1_]/np.median(flat_array_[y0_:y1_,x0_:x1_])
    
    return norm_array_


def smooth_flat_array(flat_img,amplis,kernel='mean',window_size=40,mode='mirror',percentile=1.,normalize=True,return_norm_array=False,transition=2000):
    
    if normalize==False:
        print('WARNING: running on non-normalized data. Output will contain gain information')
    '''
    if window_size%2==0:
        window_size = window_size+1
        print('ATTENTION: scipy.signal.medfilt2d does not like even numbers. Setting window_size = {0}'.format(window_size))
    print('Window size for median smoothing = {0}'.format(window_size))
    '''
    flat_array_ = flat_img.image.array
    norm_array_ = np.ones(flat_array_.shape)
    smooth_array_up = np.ones((int(flat_array_.shape[0]/2),flat_array_.shape[1]))
    smooth_array_down = np.ones((int(flat_array_.shape[0]/2),flat_array_.shape[1]))
    amplis_coords_ = get_amplis_coords(flat_img)
    
    if isinstance(amplis,str):
        sel_amplis = [amplis]
    else:
        sel_amplis = list(amplis)

    min_x0_up = []
    max_x1_up = []
    min_x0_down = []
    max_x1_down = []
    for ampli_ in sel_amplis:
        x0_ = amplis_coords_[ampli_][0]
        x1_ = amplis_coords_[ampli_][1]
        y0_ = amplis_coords_[ampli_][2]
        y1_ = amplis_coords_[ampli_][3]
        
        if normalize:
            norm_array_[y0_:y1_,x0_:x1_] = flat_array_[y0_:y1_,x0_:x1_]/np.median(flat_array_[y0_:y1_,x0_:x1_])
        
        if y1_<=transition:
            smooth_array_up[:,x0_:x1_] = np.copy(norm_array_[y0_:y1_,x0_:x1_])
            min_x0_down.append(x0_)
            max_x1_down.append(x1_)
        else:
            smooth_array_down[:,x0_:x1_] = np.copy(norm_array_[y0_:y1_,x0_:x1_])
            min_x0_up.append(x0_)
            max_x1_up.append(x1_)
        
    x0_up = np.min(np.array(min_x0_up))
    x1_up = np.max(np.array(max_x1_up))
    x0_down = np.min(np.array(min_x0_down))
    x1_down = np.max(np.array(max_x1_down))
    
    t1 = time.time()
    #smooth_array_up[:,x0_up:x1_up] = medfilt2d(smooth_array_up[:,x0_up:x1_up],kernel_size=window_size)
    #smooth_array_down[:,x0_down:x1_down] = medfilt2d(smooth_array_down[:,x0_down:x1_down],kernel_size=window_size)
    if kernel=='gauss':
        print('Smoothing with Gaussian filter')
        print('ATTENTION: window size should be equal to Gaussian standard deviation (sigma)')
        smooth_array_up[:,x0_up:x1_up] = gaussian_filter(smooth_array_up[:,x0_up:x1_up],sigma=window_size,mode=mode)
        smooth_array_down[:,x0_down:x1_down] = gaussian_filter(smooth_array_down[:,x0_down:x1_down],sigma=window_size,mode=mode)
        
    elif kernel=='mean':
        print('Smoothing with mean filter')
        
        mask_up = (smooth_array_up[:,x0_up:x1_up]>=np.percentile(smooth_array_up[:,x0_up:x1_up].flatten(),percentile))*(smooth_array_up[:,x0_up:x1_up]<=np.percentile(smooth_array_up[:,x0_up:x1_up].flatten(),100.-percentile))
        masked_array_up = np.ma.array(smooth_array_up[:,x0_up:x1_up],mask=mask_up,fill_value=-999)
        mask_down = (smooth_array_down[:,x0_down:x1_down]>=np.percentile(smooth_array_down[:,x0_down:x1_down].flatten(),percentile))*(smooth_array_down[:,x0_down:x1_down]<=np.percentile(smooth_array_down[:,x0_down:x1_down].flatten(),100.-percentile))
        masked_array_down = np.ma.array(smooth_array_down[:,x0_down:x1_down],mask=mask_down,fill_value=-999)
        
        print('Masking outliers beyond {0:.2f} and {1:.2f} percentiles'.format(percentile,100.-percentile))
        inter_array_up = np.copy(smooth_array_up)
        inter_array_down = np.copy(smooth_array_down)
        inter_array_up[:,x0_up:x1_up][~masked_array_up.mask] = np.median(smooth_array_up[:,x0_up:x1_up].flatten())
        inter_array_down[:,x0_down:x1_down][~masked_array_down.mask] = np.median(smooth_array_down[:,x0_down:x1_down].flatten())
        
        inter_array_ = np.concatenate([inter_array_up,inter_array_down],axis=0)
        
        smooth_array_up[:,x0_up:x1_up] = uniform_filter(inter_array_up[:,x0_up:x1_up],size=window_size,mode=mode)
        smooth_array_down[:,x0_down:x1_down] = uniform_filter(inter_array_down[:,x0_down:x1_down],size=window_size,mode=mode)
        
    elif kernel=='median':
        print('Smoothing with median filter')
        smooth_array_up[:,x0_up:x1_up] = median_filter(smooth_array_up[:,x0_up:x1_up],size=(window_size,window_size),mode=mode)
        smooth_array_down[:,x0_down:x1_down] = median_filter(smooth_array_down[:,x0_down:x1_down],size=(window_size,window_size),mode=mode)
    else:
        raise IOError('I do not know this kernel')
    t2 = time.time()
    print('Time for smoothing = {0:.4f}s'.format(t2-t1))
        
    smooth_array_ = np.concatenate([smooth_array_up,smooth_array_down],axis=0)
    assert smooth_array_.shape==flat_array_.shape
    
    if kernel=='mean' and return_norm_array:
        return smooth_array_, norm_array_, inter_array_ #, masked_array_up, masked_array_down
    elif kernel=='mean':
        return smooth_array_, inter_array_
    elif return_norm_array:
        return smooth_array_, norm_array_
    else:
        return smooth_array_
    


def special_flat_array(flat_img,amplis,smooth_array=None,kernel='mean',window_size=40,mode='mirror',percentile=1.,normalize=True,return_norm_array=False,transition=2000):
    
    if normalize==False:
        print('WARNING: running on non-normalized data. Output will contain gain information')
    '''
    if window_size%2==0:
        window_size = window_size+1
        print('ATTENTION: scipy.signal.medfilt2d does not like even numbers. Setting window_size = {0}'.format(window_size))
    '''
    print('Window size for {0} smoothing = {1}'.format(kernel,window_size))
    
    flat_array_ = flat_img.image.array
    #norm_array_ = np.ones(flat_array_.shape)
    special_array_ = np.ones(flat_array_.shape)
    amplis_coords_ = get_amplis_coords(flat_img)

    if isinstance(amplis,str):
        sel_amplis = [amplis]
    else:
        sel_amplis = list(amplis)
    
    inter_ = False
    if smooth_array is None:
        norm_array_ = np.ones(flat_array_.shape)
        if kernel=='mean' and return_norm_array:
            smooth_array_, norm_array_, inter_array_ = smooth_flat_array(flat_img,amplis,kernel=kernel,window_size=window_size,mode=mode,
                                                            percentile=1.,normalize=normalize,return_norm_array=return_norm_array,transition=transition)
            inter_ = True
        elif return_norm_array:
            smooth_array_, norm_array_ = smooth_flat_array(flat_img,amplis,kernel=kernel,window_size=window_size,mode=mode,
                                                            percentile=1.,normalize=normalize,return_norm_array=return_norm_array,transition=transition)
        else:
            smooth_array_ = smooth_flat_array(flat_img,amplis,kernel=kernel,window_size=window_size,mode=mode,percentile=1.,normalize=normalize,transition=transition)
        
    else:
        print('Using previously created smooth array')
        smooth_array_ = smooth_array
    
    for ampli_ in sel_amplis:
        x0_ = amplis_coords_[ampli_][0]
        x1_ = amplis_coords_[ampli_][1]
        y0_ = amplis_coords_[ampli_][2]
        y1_ = amplis_coords_[ampli_][3]

        special_array_[y0_:y1_,x0_:x1_] = flat_array_[y0_:y1_,x0_:x1_]/smooth_array_[y0_:y1_,x0_:x1_]
    
    if inter_ and return_norm_array:
        return special_array_, norm_array_, inter_array_
    elif inter_:
        return special_array_, inter_array_
    elif return_norm_array:
        return special_array_, norm_array_
    else:
        return special_array_



def plot_flat(flat_data,title=None,figsize=(10,10),cmap='gray',vmin=0.9,vmax=1.1,lognorm=False,
            butler=None,repo_embargo=True):
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)


    if isinstance(flat_data,np.ndarray):
        flat_array = flat_data
        
    elif isinstance(flat_data,int):
        if butler is None:
            raise IOError('Need to set up butler to retrieve flat')
        
        flat_array_dict = get_flat_array(flat_data,butler=butler,repo_embargo=repo_embargo)
        flat_array = flat_array_dict[flat_data]
        assert isinstance(flat_array,np.ndarray)
    
    if lognorm:
        im = ax.imshow(flat_array,cmap="gray",origin='lower',norm=LogNorm())
    else:
        im = ax.imshow(flat_array,cmap="gray",origin='lower',vmin=vmin,vmax=vmax)
    if title is not None or isinstance(flat_data,int):
        if isinstance(flat_data,int):
            title = flat_data
            print('Title set up to "{0}"'.format(flat_data))
        ax.set_title(title)

    plt.colorbar(im,ax=ax)
    
    plt.tight_layout()
    
    return 



class auxtel_flat:

    def __init__(self,flat_id,butler=None,obs_type='flat',cameraName='LATISS',detector=0,embargo=True,
                calibCollections=['LATISS/calib','LATISS/raw/all'],transition=2000):
        
        self.cameraName = cameraName
        self.detector = detector
        self.obs_type = obs_type
        
        self.flat_id = flat_id

        if butler is None:
            if embargo:
                self.repo = "/sdf/group/rubin/repo/oga/"
            else:
                self.repo = "/sdf/group/rubin/repo/main/"
           
            ####################### Butler 
            print('Butler repo: ',self.repo)
            self.butler = dafButler.Butler(self.repo)
       
        else:
            self.butler = butler
            
        self.registry = self.butler.registry
        self.calibCollections = calibCollections
        
        self.flat_img = self.butler.get(self.obs_type,instrument=self.cameraName, exposure=flat_id, detector=self.detector,collections=self.calibCollections)
        self.metadata = get_flat_metadata(flat_img=self.flat_img,butler=self.butler)
        self.flat_array = get_flat_array(flat_img=self.flat_img,butler=self.butler)
        self.dims = self.flat_array.shape
        
        self.amplis_order = [0,1,2,3,4,5,6,7,15,14,13,12,11,10,9,8]
        self.amplis_coords_all = get_amplis_coords(self.flat_img)
        self.select_amplis()
        self.transition = transition
        
        
    
    def select_amplis(self,amplis='all'):
        
        if amplis!='all':
            if isinstance(amplis,str):
                sel_amplis_ = [amplis]
            else:
                sel_amplis_ = amplis
            self.amplis = list(sel_amplis_)

            self.amplis_coords = {}
            for ampli_ in sel_amplis_:
                self.amplis_coords[ampli_] = self.amplis_coords_all[ampli_]
        else:
            self.amplis = []
            for i in range(10,18):
                self.amplis.append('{0}'.format(i))
            for i in range(8):
                self.amplis.append('0{0}'.format(7-i))
            self.amplis = np.array(self.amplis)
            self.amplis_coords = self.amplis_coords_all

        return
    
    
    def cut_flat_array(self):
        
        self.cut_array = cut_flat_array(self.flat_img,self.amplis)
        #self.cut_dims = (self.amplis_coords[])
        
        return
    
    def normalize_flat(self):
        
        self.norm_array = normalize_flat_array(self.flat_img,self.amplis)

        return
    
    
    def smooth_flat(self,kernel='mean',window_size=40,mode='mirror',normalize=True):
        if hasattr(self,'norm_array')==False:
            print('Normalized array not found. Normalizing it now')
            return_norm_array = True
        else:
            return_norm_array = False

        print('Window size for {0} smoothing = {1}'.format(kernel,window_size))
        self.window_size = window_size
        self.kernel = kernel
        if kernel=='mean' and return_norm_array:
            self.smooth_array, self.norm_array, self.inter_array = smooth_flat_array(self.flat_img,self.amplis,kernel=self.kernel,window_size=self.window_size,mode=mode,
                                                                    normalize=normalize,return_norm_array=return_norm_array,transition=self.transition)
        elif kernel=='mean':
            self.smooth_array, self.inter_array = smooth_flat_array(self.flat_img,self.amplis,kernel=self.kernel,window_size=self.window_size,mode=mode,
                                                                    normalize=normalize,transition=self.transition)
        elif return_norm_array:
            self.smooth_array, self.norm_array = smooth_flat_array(self.flat_img,self.amplis,kernel=self.kernel,window_size=self.window_size,mode=mode,
                                                                    normalize=normalize,return_norm_array=return_norm_array,transition=self.transition)
        else:
            self.smooth_array = smooth_flat_array(self.flat_img,self.amplis,kernel=self.kernel,window_size=self.window_size,mode=mode,normalize=normalize,transition=self.transition)

        return


    def special_flat(self,kernel='mean',window_size=40,mode='mirror',normalize=True):
        if hasattr(self,'norm_array')==False:
            print('Normalized array not found. Normalizing it now')
            return_norm_array = True
        else:
            return_norm_array = False

        if hasattr(self,'smooth_array')==False:
            print('ATTENTION: No smoothed flat array found. Creating it with kernel = {0} and window size = {1}'.format(kernel,window_size))
            self.window_size = window_size
            self.kernel = kernel
            '''
            if self.kernel=='mean' and return_norm_array:
                self.smooth_array, self.norm_array, self.inter_array = smooth_flat_array(self.flat_img,self.amplis,kernel=self.kernel,window_size=self.window_size,mode=mode,
                                                                                        normalize=normalize,return_norm_array=return_norm_array,return_inter_array=True,transition=self.transition)
            elif return_norm_array:
                self.smooth_array, self.norm_array = smooth_flat_array(self.flat_img,self.amplis,kernel=self.kernel,window_size=self.window_size,mode=mode,
                                                                    normalize=normalize,return_norm_array=return_norm_array,transition=self.transition)
            else:
                self.smooth_array = smooth_flat_array(self.flat_img,self.amplis,kernel=self.kernel,window_size=self.window_size,mode=mode,normalize=normalize,transition=self.transition)
            '''
        
        elif hasattr(self,'smooth_array') and (self.window_size!=window_size or self.kernel!=kernel):
            if self.window_size!=window_size:
                print('ATTENTION: Current smoothed flat has window_size = {0}. Creating it with window size = {1}'.format(self.window_size,window_size))
                self.window_size = window_size
            if self.kernel!=kernel:
                print('ATTENTION: Current smoothed flat was computed with kernel = {0}. Creating it with kernel = {1}'.format(self.kernel,kernel))
                self.kernel = kernel
            '''
            if self.kernel=='mean':
                self.smooth_array,self.inter_array = smooth_flat_array(self.flat_img,self.amplis,kernel=self.kernel,window_size=self.window_size,
                                                                        mode=mode,normalize=normalize,return_inter_array=True,transition=self.transition)
            else:
                self.smooth_array = smooth_flat_array(self.flat_img,self.amplis,kernel=self.kernel,window_size=self.window_size,mode=mode,normalize=normalize,transition=self.transition)
            '''
        self.smooth_flat(kernel=self.kernel,window_size=self.window_size,mode=mode,normalize=normalize)
        
        print('Window size for {0} smoothing = {1}'.format(self.kernel,self.window_size))
        self.special_array = special_flat_array(self.flat_img,self.amplis,smooth_array=self.smooth_array,kernel=self.kernel,
                                                window_size=self.window_size,mode=mode,normalize=normalize,transition=self.transition)
        return

    
    def plot_flat(self,show='flat',title=None,figsize=(10,10),cmap='gray',vmin=0.9,vmax=1.1,lognorm=False):
        
        if show=='flat':
            data_ = self.flat_array
            if title is not None:
                title = self.flat_id

        elif show=='cut_flat':
            if hasattr(self,'cut_array'):
                data_ = self.cut_array
                if title is not None:
                    title = title
                else:
                    title0 = 'Flat {0}, amplifiers = '.format(self.flat_id)
                    title_amplis = ''
                    for i in range(len(self.amplis)):
                        title_amplis = title_amplis+'{0}'.format(self.amplis[i])
                        if i<len(self.amplis)-1:
                            title_amplis = title_amplis+', '
                    title = title0+title_amplis

            else:
                print('ATTENTION: flat was not cut. Plotting nothing')
                return

        elif show=='norm':
            if hasattr(self,'norm_array'):
                data_ = self.norm_array
                if title is not None:
                    title = title
                else:
                    title0 = 'Norm. flat {0}, amplifiers = '.format(self.flat_id)
                    title_amplis = ''
                    for i in range(len(self.amplis)):
                        title_amplis = title_amplis+'{0}'.format(self.amplis[i])
                        if i<len(self.amplis)-1:
                            title_amplis = title_amplis+', '
                    title = title0+title_amplis

            else:
                print('ATTENTION: flat was not normalized. Plotting nothing')
                return
        
        elif show=='smooth':
            if hasattr(self,'smooth_array'):
                data_ = self.smooth_array
                if title is not None:
                    title = title
                else:
                    title0 = 'Smoothed flat {0}, amplifiers = '.format(self.flat_id)
                    title_amplis = ''
                    for i in range(len(self.amplis)):
                        title_amplis = title_amplis+'{0}'.format(self.amplis[i])
                        if i<len(self.amplis)-1:
                            title_amplis = title_amplis+', '
                    title = title0+title_amplis

            else:
                print('ATTENTION: flat was not smoothed. Plotting nothing')
                return
        
        elif show=='special':
            if hasattr(self,'special_array'):
                data_ = self.special_array
                if title is not None:
                    title = title
                else:
                    title0 = 'Special flat {0}, amplifiers = '.format(self.flat_id)
                    title_amplis = ''
                    for i in range(len(self.amplis)):
                        title_amplis = title_amplis+'{0}'.format(self.amplis[i])
                        if i<len(self.amplis)-1:
                            title_amplis = title_amplis+', '
                    title = title0+title_amplis

            else:
                print('ATTENTION: special flat was not created. Plotting nothing')
                return

        else:
            print('ATTENTION: I do not know what to plot. Plotting nothing')
            return

        plot_flat(data_,title=title,figsize=figsize,cmap=cmap,vmin=vmin,vmax=vmax,lognorm=lognorm,
                butler=self.butler)

        return



