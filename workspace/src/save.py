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

def load_data(classFolders, files_pattern, patchSize):
    """
        Loads patches into a dictionary
    """
    data = {}
    for oneClass in classFolders:
        files=glob.glob(os.path.join(oneClass,files_pattern))
        className=oneClass.split("/")[-1]        
        data[className+'_imgs']=np.zeros((len(files),patchSize,patchSize,3)).astype(np.float32)
        data[className+'_masks']=np.zeros((len(files),patchSize,patchSize,1)).astype(np.float32)
        i=0
        for file in files:
            name = os.path.splitext(os.path.basename(file))[0]
            maskFile = glob.glob(os.path.dirname(file)+"/"+name+"_mask.png")[0]
            mask = Image.open(maskFile)
            img = Image.open(file)
            imgArray = np.array(img).astype(np.float32)
            imgArray = (imgArray - imgArray.min())/(imgArray.max() - imgArray.min())
            maskArray = np.expand_dims(np.array(mask),-1).astype(np.float32)/255
            data[className+'_imgs'][i,:,:,:]=imgArray
            data[className+'_masks'][i,:,:,:]=maskArray
            i+=1
    return data

def normalizeAndSave(outputFile, data, classes, patchSize):
    """
        Normalize the data and save it in a file
    """
    j=0
    for key, val in classes.items():
        j+=data[key+"_imgs"].shape[0]
    all_data = np.zeros((j,patchSize,patchSize,3))
    i = 0
    # concatenate all the classes in one array to get the mean and std
    for key, val in classes.items():
        data[key+"_imgs"] = data[key+"_imgs"].astype(np.float32)
        data[key+"_masks"] = data[key+"_masks"].astype(np.float32)
        n = data[key+"_imgs"].shape[0] + i
        print("Writing from ", i, " to ", n, " for ", key)
        all_data[i:n,:,:,:]= data[key+"_imgs"]
        i = n
    mean = all_data.mean()
    std = all_data.std()
    
    for key, val in classes.items():
        data[key+"_imgs"] -= mean   
        data[key+"_imgs"] /= std
    
    stats = np.zeros(2)
    stats[0] = mean
    stats[1] = std
    print("Mean : ", stats[0])
    print("Std : ", stats[1])
    print("Writing in " + outputFile)
    # Create the file
    f = h5py.File(outputFile,"w")
    f.create_dataset("stats", data=stats)
    for key, val in classes.items():
        f.create_dataset(key+"_imgs", data=data[key+"_imgs"])
        f.create_dataset(key+"_masks", data=data[key+"_masks"])
    f.close()