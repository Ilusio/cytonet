import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import warnings
import glob
from PIL import Image
from keras import backend as K
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import random
try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

def transform_patch(file,j):
    """
        Transform a patch and save it
    """
    name = os.path.splitext(os.path.basename(file))[0]
    maskFile = glob.glob(os.path.dirname(file)+"/"+name+"_mask.png")[0]
    mask = Image.open(maskFile)
    img = Image.open(file)
    imgArray = np.array(img)
    maskArray = np.expand_dims(np.array(mask),-1)
    resImg, resMask = random_transform( imgArray, maskArray,
                                        rotation_range=360,
                                        width_shift_range=0.5,
                                        height_shift_range=0.5,
                                        zoom_range=0.5,
                                        shear_range=0.3,
                                        horizontal_flip = True,
                                        vertical_flip = True,
                                        fill_mode='reflect',
                                        cval=0,
                                        channel_axis = 3,
                                        row_axis = 1,
                                        col_axis = 2
                                      )
    maskImage = Image.fromarray(resMask[:,:,0])
    imgImage = Image.fromarray(resImg)
    maskImage.save(os.path.dirname(file)+"/"+name+"-" + str(j)  + "_mask.png")
    imgImage.save(os.path.dirname(file)+"/"+name+"-" + str(j)  + ".png")
    
 
def random_transform(x, y,
                    row_axis=None,
                     col_axis=None,
                     channel_axis=None,
                     rotation_range=0.,
                     height_shift_range=0.,
                     width_shift_range=0.,
                     shear_range=0.,
                     zoom_range=0.,
                     fill_mode='nearest',
                     cval=0.,
                     channel_shift_range=0.,
                     horizontal_flip=False,
                     vertical_flip=False,
                     rng=None):
        """Randomly augment a single image tensor + image mask.

        # Arguments
            x: 3D tensor, single image.
            y: 3D tensor, image mask.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        
        # random between elastic deformation or the rest 
        if(random.randint(0, 1)):
            x = np.concatenate((x, y), axis=2)
            x = elastic_transform(x,x.shape[1] * 0.8, x.shape[1] * 0.2, x.shape[1] * 0.2)
            y = x[:,:,3]
            y = np.expand_dims(y,-1)
            x = x[:,:,0:3]
        else:
            supplied_rngs = True
            if rng is None:
                supplied_rngs = False
                rng = np.random
            # x is a single image, so it doesn't have image number at index 0
            img_row_axis = row_axis - 1
            img_col_axis = col_axis - 1
            img_channel_axis = channel_axis - 1
            zoom_range = [1 - zoom_range, 1 + zoom_range]
            # use composition of homographies
            # to generate final transform that needs to be applied
            if rotation_range:
                theta = np.pi / 180 * rng.uniform(-rotation_range, rotation_range)
            else:
                theta = 0

            if height_shift_range:
                tx = rng.uniform(-height_shift_range, height_shift_range) * x.shape[img_row_axis]
            else:
                tx = 0

            if width_shift_range:
                ty = rng.uniform(-width_shift_range, width_shift_range) * x.shape[img_col_axis]
            else:
                ty = 0

            if shear_range:
                shear = rng.uniform(-shear_range, shear_range)
            else:
                shear = 0

            if zoom_range[0] == 1 and zoom_range[1] == 1:
                zx, zy = 1, 1
            else:
                zx, zy = rng.uniform(zoom_range[0], zoom_range[1], 2)

            transform_matrix = None
            if theta != 0:
                rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                            [np.sin(theta), np.cos(theta), 0],
                                            [0, 0, 1]])
                transform_matrix = rotation_matrix

            if tx != 0 or ty != 0:
                shift_matrix = np.array([[1, 0, tx],
                                         [0, 1, ty],
                                         [0, 0, 1]])
                transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

            if shear != 0:
                shear_matrix = np.array([[1, -np.sin(shear), 0],
                                        [0, np.cos(shear), 0],
                                        [0, 0, 1]])
                transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

            if zx != 1 or zy != 1:
                zoom_matrix = np.array([[zx, 0, 0],
                                        [0, zy, 0],
                                        [0, 0, 1]])
                transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

            if transform_matrix is not None:
                h, w = x.shape[img_row_axis], x.shape[img_col_axis]
                transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
                x = apply_transform(x, transform_matrix, img_channel_axis,
                                    fill_mode=fill_mode, cval=cval)
                y = apply_transform(y, transform_matrix, img_channel_axis,
                                    fill_mode=fill_mode, cval=cval)

            if channel_shift_range != 0:
                x = random_channel_shift(x,
                                         channel_shift_range,
                                         img_channel_axis)
                y = random_channel_shift(y,
                                         channel_shift_range,
                                         img_channel_axis)
            get_random = None
            if supplied_rngs:
                get_random = rng.rand
            else:
                get_random = np.random.random

            if horizontal_flip:
                if get_random()  < 0.5:
                    x = flip_axis(x, img_col_axis)
                    y = flip_axis(y, img_col_axis)

            if vertical_flip:
                if get_random()  < 0.5:
                    x = flip_axis(x, img_row_axis)
                    y = flip_axis(y, img_row_axis)
        return x, y

def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
        
    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def standardize(x, y,
                preprocessing_function=None,
                rescale=None,
                channel_axis=None,
                samplewise_center=False,
                featurewise_center=False,
                samplewise_std_normalization=False,
                featurewise_std_normalization=False,
                mean=None,
                std=None,
                zca_whitening=False,
                principal_components=None,
                rng=None):
        """Apply the normalization configuration to a batch of inputs.

        # Arguments
            x: batch of inputs to be normalized.

        # Returns
            The inputs, normalized.
        """
        if preprocessing_function:
            x = preprocessing_function(x)
        if rescale:
            x *= rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_axis = channel_axis - 1
        if samplewise_center:
            x -= np.mean(x, axis=img_channel_axis, keepdims=True)
        if samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_axis, keepdims=True) + 1e-7)

        if featurewise_center:
            if mean is not None:
                x -= mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if featurewise_std_normalization:
            if std is not None:
                x /= (std + 1e-7)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if zca_whitening:
            if principal_components is not None:
                flatx = np.reshape(x, (x.size))
                whitex = np.dot(flatx, principal_components)
                x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        return x, y

def random_rotation(x, rg, row_axis=1, col_axis=2, channel_axis=0,
                    fill_mode='nearest', cval=0.):
    """Performs a random rotation of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Rotated Numpy image tensor.
    """
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_shift(x, wrg, hrg, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shift of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Shifted Numpy image tensor.
    """
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = np.random.uniform(-hrg, hrg) * h
    ty = np.random.uniform(-wrg, wrg) * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])

    transform_matrix = translation_matrix  # no need to do offset
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_shear(x, intensity, row_axis=1, col_axis=2, channel_axis=0,
                 fill_mode='nearest', cval=0.):
    """Performs a random spatial shear of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Sheared Numpy image tensor.
    """
    shear = np.random.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_zoom(x, zoom_range, row_axis=1, col_axis=2, channel_axis=0,
                fill_mode='nearest', cval=0.):
    """Performs a random spatial zoom of a Numpy image tensor.

    # Arguments
        x: Input tensor. Must be 3D.
        zoom_range: Tuple of floats; zoom range for width and height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        Zoomed Numpy image tensor.

    # Raises
        ValueError: if `zoom_range` isn't a tuple.
    """
    if len(zoom_range) != 2:
        raise ValueError('zoom_range should be a tuple or list of two floats. '
                         'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_channel_shift(x, intensity, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.

    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x