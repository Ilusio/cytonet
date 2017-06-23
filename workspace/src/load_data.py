import glob
import os
import numpy as np
import util
from skimage import transform, io, img_as_float, exposure

"""
Data was preprocessed in the following ways:
    - resize to im_shape;
    - equalize histogram (skimage.exposure.equalize_hist);
    - expand the dimension if necessary
    - normalize by data set mean and std.
Resulting shape should be (n_samples, img_width, img_height, channels) for the image and (n_samples, img_width, img_height, 1) for the mask.
"""

def loadData(files, image_format, mask_format):
    """
        Load the data from files and preprocess it
    """
    # TODO : Exposure
    path = os.path.dirname(files[0])
    X, y = [], []
    for file in files:
        # process the image
        img = io.imread(file)
        img = transform.resize(img, (256,256,3), mode="constant")
        if(len(img)==2):
            img= np.expand_dims(img, -1)
        # progress the mask
        maskname = file.replace(image_format, mask_format) + image_format
        mask = io.imread(maskname)
        mask = transform.resize(mask, (256,256), mode="constant")
        mask = np.expand_dims(mask, -1)
        X.append(img)
        y.append(mask)
    X = np.array(X)
    y = np.array(y)
    X -= X.mean()
    X /= X.std()

    print('### Data loaded')
    print('\t{}'.format(path))
    print('\t{}\t{}'.format(X.shape, y.shape))
    print('\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max()))
    print('\tX.mean = {}, X.std = {}'.format(X.mean(), X.std()))
    return X, y

