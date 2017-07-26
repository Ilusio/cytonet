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
import time
from util import mkdirs
from skimage.util import view_as_blocks

def addBackground(imArray, maskArray):
    """
        Find the background on the array and put the value 2 on the mask
    """
    im_in = cv2.cvtColor(imArray,cv2.COLOR_BGR2GRAY)
    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.
    th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV);
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    # Remove the small parts
    kernel = np.ones((int(maskArray.shape[0]/115),int(maskArray.shape[1]/115)),np.uint8)
    opening = cv2.morphologyEx(im_out, cv2.MORPH_OPEN, kernel)
    # Invert the mask
    opening = opening.astype(np.int16)
    np.putmask(opening,opening==0,-256)
    np.putmask(opening,opening==255,0)
    opening += maskArray
    values = np.unique(opening)
    for i in values:
        if(i<0 and i>-256):
            np.putmask(opening,opening==i,i+256)
    np.putmask(opening,opening==-256,128)
    opening = opening.astype(np.uint8)
    return opening

def extractPatches(output,filename,maskname, classes, level, patchSize,j):
    """
        Extract the patches for the given file and maskname
    """
    # Opening the files
    im = op.OpenSlide(filename)
    imload = im.read_region((0,0), level, im.level_dimensions[level])
    print("Image dimension : ", im.level_dimensions[level])
    mask = Image.open(maskname)
    if(imload.size != mask.size):
       mask = mask.resize(imload.size)
    imArray = np.array(imload)
    maskArray = np.array(mask)
    halfPatch = patchSize//2
    
    #Preprocess
    maskArray_back = addBackground(imArray, maskArray)
    imArray = np.lib.pad(imArray[:,:,0:3], ((patchSize-(imArray.shape[0]%patchSize), 0), (patchSize-(imArray.shape[1]%patchSize), 0),(0,0)), 'reflect')
    maskArrayPad = np.lib.pad(maskArray_back, ((patchSize-maskArray_back.shape[0]%patchSize, 0), (patchSize-maskArray_back.shape[1]%patchSize, 0)), 'reflect')
    np.putmask(maskArrayPad, maskArrayPad==1, 255)
    images_array = view_as_blocks(imArray, (64,64,3))
    images_array = images_array.reshape(-1,64,64,3)
    masks_array = view_as_blocks(maskArrayPad, (64,64))
    masks_array = masks_array.reshape(-1,64,64)
    for i in range(masks_array.shape[0]):    
        imageName = output + "/image_" + str(j) + ".png"
        imageNameMask =  output + "/image_" + str(j) +"_mask.png"
        misc.imsave(imageName,images_array[i])
        misc.imsave(imageNameMask,masks_array[i])
        os.chmod(imageName , 0o777)
        os.chmod(imageNameMask, 0o777)
        j+=1
        if(j%100==0):
            print("",j," patches extracted")
    return j

def extractFiles(files,
            outputFolder, 
            maskPattern, 
            classes, 
            level, 
            patchSize):
    """
        Extract all the files of a folder
    """
    j = 0
    for oneFile in files:
        name = os.path.splitext(os.path.basename(oneFile))[0]
        mkdirs(outputFolder,0o777)
        print("Extracting " + name)
        maskFile = glob.glob(os.path.dirname(oneFile)+"/"+name+maskPattern)[0]
        j = extractPatches(outputFolder, 
                           oneFile, 
                           maskFile,  
                           classes,  
                           level, 
                           patchSize,
                           j)
        print("Extraction for ", name, " finished")