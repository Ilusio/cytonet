from PIL import Image, ImageDraw, ImageFont
import numpy as np
from skimage import measure
import time
import gc
from matplotlib import pyplot as plt
from scipy import misc
import openslide as op
import configparser
import glob
import ast
import errno
import os
from util import mkdirs, extend_glob, file_suffix


def evaluate(prediction, filename, output_folder,load_level, transparency, display):
    FN = 0 # Number of false negatives
    TP = 0 # Number of true positives
    FP = 0 # Number of false positives
    
    # Get the data
    mask = Image.open(prediction)
    mask = mask.convert("RGBA")
    maskArray = np.array(mask).astype(np.uint16) # uint8 might not be enough to count connected components
    maskArray[-(maskArray.shape[0]%256):,:,0:3]=0
    maskArray[:,-(maskArray.shape[1]%256):,0:3]=0

    # Changing the channels (R and B are normally prediction and G ground truth)
    maskArray[:,:,2]=maskArray[:,:,1]
    maskArray[:,:,1]=maskArray[:,:,0]
    maskArray[:,:,0]=0

    # Creating pointing variable for readibility 
    pred = maskArray[:,:,1]
    fp_array = maskArray[:,:,0]
    gt = maskArray[:,:,2]

    # Finding the connected components
    maskArray[:,:,1] = measure.label(pred)
    pred_nb = pred.max()
    maskArray[:,:,2] = measure.label(gt)
    gt_nb = gt.max()
    pred_labels = measure.regionprops(maskArray[:,:,1])
    gt_labels = measure.regionprops(maskArray[:,:,2])

    # Initializing arrays with the connected parts coordinates
    bb_pred = np.zeros((pred_nb,4), dtype=np.uint32)
    bb_gt = np.zeros((gt_nb,4), dtype=np.uint32)
    # Filling the arrays with coordinate
    for i in range(len(pred_labels)):
        bb_pred[pred_labels[i].label-1][0] = pred_labels[i].bbox[0]
        bb_pred[pred_labels[i].label-1][1] = pred_labels[i].bbox[2]
        bb_pred[pred_labels[i].label-1][2] = pred_labels[i].bbox[1]
        bb_pred[pred_labels[i].label-1][3] = pred_labels[i].bbox[3]

    for i in range(len(gt_labels)):
        bb_gt[gt_labels[i].label-1][0] = gt_labels[i].bbox[0]
        bb_gt[gt_labels[i].label-1][1] = gt_labels[i].bbox[2]
        bb_gt[gt_labels[i].label-1][2] = gt_labels[i].bbox[1]
        bb_gt[gt_labels[i].label-1][3] = gt_labels[i].bbox[3]

    # Display the result
    if(display):
        plt.figure(figsize=(50,50))
        plt.subplot(131)
        plt.imshow(pred, cmap='nipy_spectral')
        plt.subplot(132)
        plt.imshow(gt, cmap='nipy_spectral')
        plt.show()
        
    # Check if our prediction are FP or TP
    gt_TP_indices = [] # will contains the indexes of connected component that match a TP#
    for i in range(1,pred_nb+1):
        pred_ind = i-1
        gt_indices = gt[bb_pred[pred_ind][0]:bb_pred[pred_ind][1],bb_pred[pred_ind][2]:bb_pred[pred_ind][3]]
        gt_indices = np.unique(gt_indices)
        # No matching prediction
        if(len(gt_indices)==1 and gt_indices[0]==0):
            FP += 1
            fp_array[bb_pred[pred_ind][0]:bb_pred[pred_ind][1],bb_pred[pred_ind][2]:bb_pred[pred_ind][3]]=255
            pred[bb_pred[pred_ind][0]:bb_pred[pred_ind][1],bb_pred[pred_ind][2]:bb_pred[pred_ind][3]]=0
        else:
            # remove the 0 as an index
            if(gt_indices[0]==0):
                gt_indices=np.delete(gt_indices,0)
            max_overlap = -1 # the maximum overlap for all the indexes of the max
            max_indice = -1 # the corresponding index
            for ind in gt_indices:
                gt_ind = ind -1
                # Get the coordinate of the intersection
                #print("Checking ", pred_ind, " and ", gt_ind)
                #print("Sizes are ", bb_pred.shape, " and ", bb_gt.shape)
                xmin = max(bb_pred[pred_ind][0],bb_gt[gt_ind][0])
                xmax = min(bb_pred[pred_ind][1],bb_gt[gt_ind][1])
                ymin = max(bb_pred[pred_ind][2],bb_gt[gt_ind][2])
                ymax = min(bb_pred[pred_ind][3],bb_gt[gt_ind][3])
                # Overlap calculation
                SI = (xmax - xmin)*(ymax- ymin)
                SA = (bb_pred[pred_ind][1]-bb_pred[pred_ind][0])*(bb_pred[pred_ind][3]-bb_pred[pred_ind][2])
                SB = (bb_gt[gt_ind][1]-bb_gt[gt_ind][0])*(bb_gt[gt_ind][3]-bb_gt[gt_ind][2])
                SU = SA + SB - SI
                overlap = SI / SU
                if(overlap>max_overlap):
                    max_overlap=overlap
                    max_indice=gt_ind
            # If the overlap rate is above 20% we consider it as a TP
            if(max_overlap>0.2):
                TP += 1
                pred[bb_pred[pred_ind][0]:bb_pred[pred_ind][1],bb_pred[pred_ind][2]:bb_pred[pred_ind][3]]=255
                gt_TP_indices.append(max_indice)
            else:
                FP += 1
                fp_array[bb_pred[pred_ind][0]:bb_pred[pred_ind][1],bb_pred[pred_ind][2]:bb_pred[pred_ind][3]]=255
                pred[bb_pred[pred_ind][0]:bb_pred[pred_ind][1],bb_pred[pred_ind][2]:bb_pred[pred_ind][3]]=0
    bb_gt = np.delete(bb_gt,gt_TP_indices,0) # remove the TP components from the mask

    # Reset the ground truth array
    maskArray[:,:,2] = 0

    # Finding the FN
    for i in range(0,bb_gt.shape[0]):
        FN += 1
        gt[bb_gt[i][0]:bb_gt[i][1],bb_gt[i][2]:bb_gt[i][3]] = 255

    maskArray = maskArray.astype(np.uint8)

    # Display the result
    if(display):
        plt.figure(figsize=(50,50))
        plt.subplot(131)
        plt.imshow(pred, cmap='Greens')
        plt.subplot(132)
        plt.imshow(fp_array, cmap='Reds')
        plt.subplot(133)
        plt.imshow(gt, cmap='Blues')
        plt.show()
        
    # Display the result
    if(display):
        plt.figure(figsize=(50,50))
        plt.subplot(131)
        plt.imshow(maskArray)
        plt.show()

    # Opening the original image
    im = op.OpenSlide(filename)
    imload = im.read_region((0,0), load_level, im.level_dimensions[load_level])

    # Adding transparency on the mask
    maskArray[:,:,3]=0
    maskArray[:,:,3] += maskArray[:,:,0] + maskArray[:,:,1] + maskArray[:,:,2]
    np.putmask(maskArray[:,:,3],maskArray[:,:,3]!=0,transparency)

    # Calculate the Fscore
    Fscore = (2*TP)/(2*TP+FP+FN)
    recall = (TP)/(TP+FN)
    precision = 0 if (TP==0 and FP==0) else (TP)/(TP+FP)
    print("Fscore : ", Fscore)
    print("TP : ", TP)
    print("FP : ", FP)
    print("FN : ", FN)

    filename_base = prediction.split("/")[-1].split(".png")[0]

    output_image = os.path.join(output_folder , filename_base + "_" + str(round(Fscore*100,2)) +".png")

    # Merge the mask and image and save it
    maskImage = Image.fromarray(maskArray, 'RGBA')
    maskImage = Image.alpha_composite(imload, maskImage)
    maskImage = maskImage.crop((0,0,maskImage.size[0], maskImage.size[1]+150))
    
    # Add caption
    draw = ImageDraw.Draw(maskImage)
    draw.rectangle(((0,maskImage.size[1]-150), (maskImage.size[0],maskImage.size[1])), fill=(255,255,255))
    draw.rectangle(((15, maskImage.size[1]-65), (65, maskImage.size[1]-15)), fill=(255,0,0))
    draw.rectangle(((maskImage.size[0]/3 +15, maskImage.size[1]-65), (maskImage.size[0]/3 +65, maskImage.size[1]-15)), fill=(0,255,0))
    draw.rectangle(((maskImage.size[0]*2/3 +15, maskImage.size[1]-65), (maskImage.size[0]*2/3 + 65, maskImage.size[1]-15)), fill=(0,0,255))
    
    # Add caption text
    font = ImageFont.truetype("./font/Roboto-Medium.ttf", 40)
    draw.text((80, maskImage.size[1]-63),"False positive : " + str(FP),(0,0,0),font=font)
    draw.text((maskImage.size[0]/3 + 80, maskImage.size[1]-63),"True positive : " + str(TP),(0,0,0),font=font)
    draw.text((maskImage.size[0]*2/3 + 80, maskImage.size[1]-63),"False positive : " + str(FN),(0,0,0),font=font)
    draw.text((15, maskImage.size[1]-135),"F-score : " + str(round(Fscore, 5)),(0,0,0),font=font)
    draw.text((maskImage.size[0]/3 +15, maskImage.size[1]-135),"Recall : " + str(round(recall, 5)),(0,0,0),font=font)
    draw.text((maskImage.size[0]*2/3 +15, maskImage.size[1]-135),"Precision : " + str(round(precision, 5)),(0,0,0),font=font)
    maskImage.save(output_image)