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

def initValues(files, level, maskPattern, classes):
    extracting = {} # classes with connected components
    none_extracting = {} # the number of patches of this class depends on the previous dictionnary
    max_extraction = 0 # the number of patches for the none classes
    n_files = len(files)
    for key, tup in classes.items():
        val = tup[1]
        if(val==None):
            none_extracting[key]=0
        elif(val<=0):
            extracting[key]=0
    # classes cropped by connected components
    if(len(extracting)>0):
        for file in files:
            # reading the images
            print("Working on " + file)
            name = os.path.splitext(os.path.basename(file))[0]
            maskname = glob.glob(os.path.dirname(file)+"/"+name+maskPattern)[0]
            im = op.OpenSlide(file)
            mask = Image.open(maskname)
            if(im.level_dimensions[level] != mask.size):
               mask = mask.resize(im.level_dimensions[level])
            maskArray = np.array(mask)
            del im
            # find the number of connected components
            for key, val in extracting.items():
                nb_patches = (0 - classes[key][1]) + 1 # number of shifts
                # find connected components
                maskClass = np.array(maskArray)
                np.putmask(maskClass,maskClass!=classes[key][0],0)
                maskClass = measure.label(maskClass)
                nb_extract = maskClass.max()
                # take max images or the number of connected components               
                try:
                    max_class=classes[key][2]
                    if(max_class<maskClass.max()):
                        print("Found ", maskClass.max(), " components for ", key, " but max number is ", max_class)
                        nb_extract = max_class
                except:
                    pass
                extracting[key] += nb_extract * nb_patches
                print("New number of extraction for " , key, " : ", extracting[key])
        # initialize the arrays
        for key, val in extracting.items():
            if(val>max_extraction):
                max_extraction=val
    # initialize the arrays for none classes
    for key, val in none_extracting.items():
        classes[key]=(classes[key][0],int(int(max_extraction/n_files)*classes[key][2]),classes[key][2])
        print("Number of extraction per file for ", key, " is ", classes[key][1])
    return classes

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

def extractPatches(output,filename,maskname, classes, level, patchSize,j, background):
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
    extract_mask = maskArray_back if background else maskArray
    imArray = np.lib.pad(imArray, ((halfPatch, halfPatch), (halfPatch, halfPatch),(0,0)), 'reflect')
    maskArrayPad = np.lib.pad(extract_mask, ((halfPatch, halfPatch), (halfPatch, halfPatch)), 'reflect')
    np.putmask(maskArrayPad, maskArrayPad==1, 255)
    # Extraction
    for key, tup in classes.items():
        print("Extracting patches for ", key)
        # classes with the number of patches specified
        img_class = tup[1]
        if(img_class>0):
            print("Extracting ", img_class, " patches ")
            indices = np.where(maskArray_back==tup[0])
            sample = random.sample(range(len(indices[0])), img_class)
            maskClass = np.array(maskArrayPad) #TODO : remove this ?  
            for i in sample:
                x=indices[0][i]
                y=indices[1][i]
                x2 = x+patchSize
                y2 = y+patchSize
                croppedIm = imArray[x:x2,y:y2,0:3]
                croppedMask = maskClass[x:x2,y:y2]              
                # create the images
                imageName = output + "/" + key + "/image_" + str(j) + ".png"
                imageNameMask =  output + "/" + key + "/image_" + str(j) +"_mask.png"
                misc.imsave(imageName,croppedIm)
                misc.imsave(imageNameMask,croppedMask)
                os.chmod(imageName , 0o777)
                os.chmod(imageNameMask, 0o777)
                j+=1
                if(j%100==0):
                    print("",j," patches extracted")
        else:
            # classes with connected components
            nb_patches = (0 - tup[1]) + 1
            maskClass = np.array(maskArray_back)
            np.putmask(maskClass,maskClass!=tup[0],0)
            maskClass = measure.label(maskClass)
            bb_labels = measure.regionprops(maskClass)
            nb_extract = maskClass.max()
            extract_sample = range(len(bb_labels))
            try:
                max_class= tup[2]
                if(max_class<maskClass.max()):
                    nb_extract = max_class
                    extract_sample = random.sample(extract_sample,nb_extract)
            except:
                pass  
            for i in extract_sample:
                bb = bb_labels[i].bbox
                x_center = int((bb[0]+bb[2]) / 2) + halfPatch
                y_center = int((bb[1]+bb[3]) / 2) + halfPatch
                division = 2
                shifting = [(0,-patchSize/division),(-patchSize/division,0),(0,patchSize/division),(patchSize/division,0), \
                            (-patchSize/division,-patchSize/division), (-patchSize/division,patchSize/division), \
                            (patchSize/division,patchSize/division), (patchSize/division,-patchSize/division)
                           ]
                # shifting
                for h in range(0,nb_patches):
                    x = x_center
                    y = y_center
                    if(h==0):
                        None
                    else:
                        if(len(shifting)>0):
                            rd = random.randint(0, len(shifting)-1)
                            shift = shifting[rd]
                            x+=shift[0]
                            y+=shift[1]
                            del shifting[rd]
                        else:
                            x += random.randint(-patchSize/division,patchSize/division)     
                            y += random.randint(-patchSize/division,patchSize/division)
                    x1 = int(x-patchSize/2)
                    x2 = int(x+patchSize/2)
                    y1 = int(y-patchSize/2)
                    y2 = int(y+patchSize/2)
                    # cropping
                    croppedIm = imArray[x1:x2,y1:y2,0:3]
                    croppedMask = maskArrayPad[x1:x2,y1:y2]
                    # create the images if needed
                    imageName = output + "/" + key + "/image_" + str(j) + ".png"
                    imageNameMask =  output + "/" + key + "/image_" + str(j) +"_mask.png"
                    misc.imsave(imageName,croppedIm)
                    misc.imsave(imageNameMask,croppedMask)
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
            patchSize,
            background):
    """
        Extract all the files of a folder
    """
    j = 0
    new_classes = initValues(files, level, maskPattern, classes)
    for oneFile in files:
        name = os.path.splitext(os.path.basename(oneFile))[0]
        for key, val in classes.items():
            folder = os.path.join(outputFolder,key)
            mkdirs(folder,0o777)
        print("Extracting " + name)
        maskFile = glob.glob(os.path.dirname(oneFile)+"/"+name+maskPattern)[0]
        j = extractPatches(outputFolder, 
                           oneFile, 
                           maskFile,  
                           new_classes,  
                           level, 
                           patchSize,
                           j,
                           background)
        print("Extraction for ", name, " finished")