from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dropout


def build_UNet(inp_shape, nb_classes, k_size=3):
    """
        Build the unet model
    """
    
    merge_axis = -1 # Feature maps are concatenated along last axis (for tf backend, should be 0 for theano)
    
    data = Input(shape=inp_shape)
    
    conv1 = Convolution2D(filters=(64), kernel_size=k_size, padding='same', activation='relu')(data)
    conv1 = Convolution2D(filters=(64), kernel_size=k_size, padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(filters=(128), kernel_size=k_size, padding='same', activation='relu')(pool1)
    conv2 = Convolution2D(filters=(128), kernel_size=k_size, padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(filters=(256), kernel_size=k_size, padding='same', activation='relu')(pool2)
    conv3 = Convolution2D(filters=(256), kernel_size=k_size, padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(filters=(512), kernel_size=k_size, padding='same', activation='relu')(pool3)
    conv4 = Convolution2D(filters=(512), kernel_size=k_size, padding='same', activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(filters=(1024), kernel_size=k_size, padding='same', activation='relu')(pool4)

    up1 = UpSampling2D(size=(2, 2))(conv5)
    merged1 = concatenate([up1,conv4], axis=merge_axis)
    conv6 = Convolution2D(filters=(512), kernel_size=k_size, padding='same', activation='relu')(merged1)
    conv6 = Convolution2D(filters=(512), kernel_size=k_size, padding='same', activation='relu')(conv6)

    up2 = UpSampling2D(size=(2, 2))(conv6)
    merged2 = concatenate([up2,conv3], axis=merge_axis)
    conv7 = Convolution2D(filters=(256), kernel_size=k_size, padding='same', activation='relu')(merged2)
    conv7 = Convolution2D(filters=(256), kernel_size=k_size, padding='same', activation='relu')(conv7)

    up3 = UpSampling2D(size=(2, 2))(conv7)
    merged3 = concatenate([up3,conv2], axis=merge_axis)
    conv8 = Convolution2D(filters=(128), kernel_size=k_size, padding='same', activation='relu')(merged3)
    conv8 = Convolution2D(filters=(128), kernel_size=k_size, padding='same', activation='relu')(conv8)

    up4 = UpSampling2D(size=(2, 2))(conv8)
    merged4 = concatenate([up4,conv1], axis=merge_axis)
    conv9 = Convolution2D(filters=(64), kernel_size=k_size, padding='same', activation='relu')(merged4)
    conv9 = Convolution2D(filters=(64), kernel_size=k_size, padding='same', activation='relu')(conv9)

    conv10 = Convolution2D(filters=nb_classes, kernel_size=1, padding='same', activation='sigmoid')(conv9)

    output = conv10
    model = Model(data, output)
    return model

def build_UNet_deconv(inp_shape, nb_classes, k_size=3):
    """
        Build the unet model
    """
    # TODO : Number of channels in parameters
    
    merge_axis = -1 # Feature maps are concatenated along last axis (for tf backend, should be 0 for theano)
    
    data = Input(shape=inp_shape)
    
    conv1 = Convolution2D(filters=(64), kernel_size=k_size, padding='same', activation='relu')(data)
    conv1 = Convolution2D(filters=(64), kernel_size=k_size, padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(filters=(128), kernel_size=k_size, padding='same', activation='relu')(pool1)
    conv2 = Convolution2D(filters=(128), kernel_size=k_size, padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(filters=(256), kernel_size=k_size, padding='same', activation='relu')(pool2)
    conv3 = Convolution2D(filters=(256), kernel_size=k_size, padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(filters=(512), kernel_size=k_size, padding='same', activation='relu')(pool3)
    conv4 = Convolution2D(filters=(512), kernel_size=k_size, padding='same', activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(filters=(1024), kernel_size=k_size, padding='same', activation='relu')(pool4)

    #up1 = UpSampling2D(size=(2, 2))(conv5)
    up1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    merged1 = concatenate([conv4, up1], axis=merge_axis)
    conv6 = Convolution2D(filters=(512), kernel_size=k_size, padding='same', activation='relu')(merged1)
    conv6 = Convolution2D(filters=(512), kernel_size=k_size, padding='same', activation='relu')(conv6)

    #up2 = UpSampling2D(size=(2, 2))(conv6)
    up2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    merged2 = concatenate([conv3, up2], axis=merge_axis)
    conv7 = Convolution2D(filters=(256), kernel_size=k_size, padding='same', activation='relu')(merged2)
    conv7 = Convolution2D(filters=(256), kernel_size=k_size, padding='same', activation='relu')(conv7)

    #up3 = UpSampling2D(size=(2, 2))(conv7)
    up3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    merged3 = concatenate([conv2, up3], axis=merge_axis)
    conv8 = Convolution2D(filters=(128), kernel_size=k_size, padding='same', activation='relu')(merged3)
    conv8 = Convolution2D(filters=(128), kernel_size=k_size, padding='same', activation='relu')(conv8)

    #up4 = UpSampling2D(size=(2, 2))(conv8)
    up4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    merged4 = concatenate([conv1, up4], axis=merge_axis)
    conv9 = Convolution2D(filters=(64), kernel_size=k_size, padding='same', activation='relu')(merged4)
    conv9 = Convolution2D(filters=(64), kernel_size=k_size, padding='same', activation='relu')(conv9)

    conv10 = Convolution2D(filters=nb_classes, kernel_size=1, padding='same', activation='sigmoid')(conv9)

    output = conv10
    model = Model(data, output)
    return model

def build_UNet_dropout(inp_shape, nb_classes, k_size=3):
    """
        Build the unet model
    """
    # TODO : Number of channels in parameters
    
    merge_axis = -1 # Feature maps are concatenated along last axis (for tf backend, should be 0 for theano)
    drop_rate = 0.5
    
    data = Input(shape=inp_shape)
    
    conv1 = Convolution2D(filters=(64), kernel_size=k_size, padding='same', activation='relu')(data)
    conv1 = Dropout(drop_rate)(conv1)
    conv1 = Convolution2D(filters=(64), kernel_size=k_size, padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(filters=(128), kernel_size=k_size, padding='same', activation='relu')(pool1)
    conv2 = Dropout(drop_rate)(conv2)
    conv2 = Convolution2D(filters=(128), kernel_size=k_size, padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(filters=(256), kernel_size=k_size, padding='same', activation='relu')(pool2)
    conv3 = Dropout(drop_rate)(conv3)
    conv3 = Convolution2D(filters=(256), kernel_size=k_size, padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(filters=(512), kernel_size=k_size, padding='same', activation='relu')(pool3)
    conv4 = Dropout(drop_rate)(conv4)
    conv4 = Convolution2D(filters=(512), kernel_size=k_size, padding='same', activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(filters=(1024), kernel_size=k_size, padding='same', activation='relu')(pool4)

    up1 = UpSampling2D(size=(2, 2))(conv5)
    merged1 = concatenate([up1,conv4], axis=merge_axis)
    conv6 = Convolution2D(filters=(512), kernel_size=k_size, padding='same', activation='relu')(merged1)
    conv6 = Dropout(drop_rate)(conv6)
    conv6 = Convolution2D(filters=(512), kernel_size=k_size, padding='same', activation='relu')(conv6)

    up2 = UpSampling2D(size=(2, 2))(conv6)
    merged2 = concatenate([up2,conv3], axis=merge_axis)
    conv7 = Convolution2D(filters=(256), kernel_size=k_size, padding='same', activation='relu')(merged2)
    conv7 = Dropout(drop_rate)(conv7)
    conv7 = Convolution2D(filters=(256), kernel_size=k_size, padding='same', activation='relu')(conv7)

    up3 = UpSampling2D(size=(2, 2))(conv7)
    merged3 = concatenate([up3,conv2], axis=merge_axis)
    conv8 = Convolution2D(filters=(128), kernel_size=k_size, padding='same', activation='relu')(merged3)
    conv8 = Dropout(drop_rate)(conv8)
    conv8 = Convolution2D(filters=(128), kernel_size=k_size, padding='same', activation='relu')(conv8)

    up4 = UpSampling2D(size=(2, 2))(conv8)
    merged4 = concatenate([up4,conv1], axis=merge_axis)
    conv9 = Convolution2D(filters=(64), kernel_size=k_size, padding='same', activation='relu')(merged4)
    conv9 = Dropout(drop_rate)(conv9)
    conv9 = Convolution2D(filters=(64), kernel_size=k_size, padding='same', activation='relu')(conv9)

    conv10 = Convolution2D(filters=nb_classes, kernel_size=1, padding='same', activation='sigmoid')(conv9)

    output = conv10
    model = Model(data, output)
    return model

def build_UNet_lung(inp_shape, nb_classes, k_size=3):
    """
        Build the unet model
    """
    # TODO : Number of channels in parameters
    
    merge_axis = -1 # Feature maps are concatenated along last axis (for tf backend, should be 0 for theano)
    
    data = Input(shape=inp_shape)
    
    conv1 = Convolution2D(filters=32, kernel_size=k_size, padding='same', activation='relu')(data)
    conv1 = Convolution2D(filters=32, kernel_size=k_size, padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(pool1)
    conv2 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(pool2)
    conv3 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(pool3)
    conv4 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(pool4)

    up1 = UpSampling2D(size=(2, 2))(conv5)
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(up1)
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(conv6)
    merged1 = concatenate([conv4, conv6], axis=merge_axis)
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(merged1)

    up2 = UpSampling2D(size=(2, 2))(conv6)
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(up2)
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(conv7)
    merged2 = concatenate([conv3, conv7], axis=merge_axis)
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(merged2)

    up3 = UpSampling2D(size=(2, 2))(conv7)
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(up3)
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(conv8)
    merged3 = concatenate([conv2, conv8], axis=merge_axis)
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(merged3)

    up4 = UpSampling2D(size=(2, 2))(conv8)
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(up4)
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv9)
    merged4 = concatenate([conv1, conv9], axis=merge_axis)
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(merged4)

    conv10 = Convolution2D(filters=nb_classes, kernel_size=1, padding='same', activation='sigmoid')(conv9)

    output = conv10
    model = Model(data, output)
    return model