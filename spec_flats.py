import os
import matplotlib.pyplot as plt
import re

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.ticker                         # here's where the formatter is
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from astropy.io import fits
from astropy.time import Time

from scipy.optimize import curve_fit
import scipy.ndimage as ndimage
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



def find_closest_date(date,flat_dates,flat_ids):

    obs_day = int(date)
    
    print('Requested observation date = ', obs_day)
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
        cut_array = np.ones(flat_array_.shape)
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

            cut_array[y0_:y1_,x0_:x1_] = flat_array_[y0_:y1_,x0_:x1_]
            
        return cut_array


def normalize_flat_array(flat_img,amplis):
    
    flat_array_ = flat_img.image.array
    norm_array = np.ones(flat_array_.shape)
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
        
        norm_array[y0_:y1_,x0_:x1_] = flat_array_[y0_:y1_,x0_:x1_]/np.median(flat_array_[y0_:y1_,x0_:x1_])
    
    return norm_array


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
            self.amplis_coords = self.amplis_coords_all

        return
    
    
    def cut_flat_array(self):
        
        self.cut_array = cut_flat_array(self.flat_img,self.amplis)

        return
        '''
        if self.amplis=='all':
            print('All amplifiers are selected. Nothing to cut')
            return 
        else:
            cut_array = np.ones(self.flat_array.shape) 
            for ampli_ in self.amplis:
                x0_ = self.amplis_coords[ampli_][0]
                x1_ = self.amplis_coords[ampli_][1]
                y0_ = self.amplis_coords[ampli_][2]
                y1_ = self.amplis_coords[ampli_][3]

                cut_array[y0_:y1_,x0_:x1_] = self.flat_array[y0_:y1_,x0_:x1_]
            self.cut_array = cut_array

            return
        '''
    
    def normalize_flat_array(self):
        
        self.norm_array = normalize_flat_array(self.flat_img,self.amplis)

        return
        '''
        norm_array = np.ones(self.flat_array.shape)
        for i,ampli_ in enumerate(self.amplis):
            x0_ = self.amplis_coords[ampli_][0]
            x1_ = self.amplis_coords[ampli_][1]
            y0_ = self.amplis_coords[ampli_][2]
            y1_ = self.amplis_coords[ampli_][3]
            
            norm_array[y0_:y1_,x0_:x1_] = self.flat_array[y0_:y1_,x0_:x1_]/np.median(self.flat_array[y0_:y1_,x0_:x1_])
        
        self.norm_array = norm_array
        
        return
        '''
        
    
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
        

        plot_flat(data_,title=title,figsize=figsize,cmap=cmap,vmin=vmin,vmax=vmax,lognorm=lognorm,
                butler=self.butler)

        return



