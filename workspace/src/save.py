import openslide as op
from PIL import Image
import numpy as np
import random
import glob
import os
import util
import h5py
import cv2
from skimage import measure
from matplotlib import pyplot as plt
from scipy import misc, ndimage
from skimage import morphology
from skimage import color
from skimage import io

def load_data(files, files_pattern, patchSize):
    """
        Loads patches into a dictionary
    """
    data = {}    
    data['imgs']=np.zeros((len(files),patchSize,patchSize,3)).astype(np.float32)
    data['masks']=np.zeros((len(files),patchSize,patchSize,1)).astype(np.float32)
    i=0
    for file in files:
            name = os.path.splitext(os.path.basename(file))[0]
            maskFile = glob.glob(os.path.dirname(file)+"/"+name+"_mask.png")[0]
            mask = Image.open(maskFile)
            img = Image.open(file)
            imgArray = np.array(img).astype(np.float32)/255
            maskArray = np.expand_dims(np.array(mask),-1).astype(np.float32)
            np.putmask(maskArray,maskArray==128,2)
            np.putmask(maskArray,maskArray==255,1)
            data['imgs'][i,:,:,:]=imgArray
            data['masks'][i,:,:,:]=maskArray
            i+=1
    return data

def normalizeAndSave(outputFile, data, classes, patchSize):
    """
        Normalize the data and save it in a file
    """
    all_data = data['imgs']
    mean = all_data.mean()
    std = all_data.std()
    all_data -= mean   
    all_data /= std
    
    stats = np.zeros(2)
    stats[0] = mean
    stats[1] = std
    print("Mean : ", stats[0])
    print("Std : ", stats[1])
    print("Writing in " + outputFile)
    # Create the file
    f = h5py.File(outputFile,"w")
    f.create_dataset("stats", data=stats)
    f.create_dataset("imgs", data=all_data)
    f.create_dataset("masks", data=data["masks"])
    f.close()