# Sources

## Pipeline

![avemaria](https://user-images.githubusercontent.com/9282351/28579973-72a1f63c-715e-11e7-895d-aa7d261d1b6c.png)

#### Extraction

This step extracts the patches of differents classes from files. The output is a folder with the patches.

#### DataAugmentation

**This step is currently not working. The interpolation on the masks creates new values for the pixels and break the code of the training.**
This is optionnal. You can generate more patches thanks to this step. For example, this will had some elastic deformation, zoom, rotation ... You will get more patches in the input folder.

#### SaveAndNorm

This step normalizes the patches and then saves them as a matrice file for the training.

#### Training

This step trains a network with the given patches.

#### Segmentation

The segmentation uses the trained network and gives a heatmap as an output.

#### Evaluation

The evaluation uses the prediction from the segmentation and the original mask and computes the F1-score, the recall and precision and provide an images with the true positive, false positive and false negative. 

## Configuration

The configuration file is named cytonet.cfg. Each step of the pipeline has a section in it plus a general one. You need to modify this file before launching any script.

#### General
- patch_size : Size of the patches.
- load_level : Parameter for SVS file. Define the loading level for the image.

#### Extraction
- filenames : Input files for the extraction. This must be an array of globs. Ex : ['/root/workspace/data/14_02/*.svs', '/root/workspace/data/10_02/*.svs']
- output_folder : Output folder for the patches.
- mask_pattern : Mask pattern for the filemaes. For example, if the filename is IFTA_10_02.svs and the maskname is IFTA_10_02_mask.png, the pattern should be something like *_mask.png
- classes : Dictionary of tuples giving the value/number of patches/max patches. For example : {'neg': (0,None,2), 'pos' : (255,0), 'back' : (128,20)}. The first member of the tuple is the value of the pixels on the mask. If the second member is greater than 0, it will extract this number of patches randomly on the image. If it's equal to 0, it will extract one patch per connected component. If it's less than 0, it will extract one patch centered on each component plus -n+1 shiftings on each one (ie if you have 100 components and you choose -1 you will get 200 patches). You can also put None as the second member then you have to add a third parameter. This last parameter will take the maximum number of patches for all the classes and multiply it by the parameter. In the exemple, if you have 500 pos patches, you will get 1000 neg patches.  
- load_level : Parameter for SVS file. Define the loading level for the image. If not defined, the parameter load_level in the general section is used.
- patch_size : Size of the patches. If not defined, the parameter patch_size in the general section is used.

#### DataAugmentation
- input_folder : Input folder for the data augmentation. If not defined, the parameter out_folder in the section extraction is used.
- files_pattern : Glob for the filenames. It must exclude the masks. (if the maskname ends with mask you can use something like *![mask].png).
- classes : Dictionary with the classes as the keys and the number of patches you wish to get as the value. For example : {'pos':-4, 'neg':2000}. If the number is greater than 0, patches will be generated until this number is reached. If the number is less than 0, the code will multiply the number of patches by this number and generate this amount of patches.

#### SaveAndNorm
- input_folder : Input folder containing the patches. If not defined, the parameter output_folder in the extraction section is used.
- output_file : Output file as a matrice containing the patches.
- files_pattern : Glob for the filenames. It must exclude the masks. (if the maskname ends with mask you can use something like *![mask].png).
- classes : Dictionary with the classes as the keys and their pixel value as the values.
- patch_size : Size of the patches. If not defined, the parameter patch_size in the general section is used.

#### Training
- valid_portion : Percentage of the data used for validation. Must be between 0 and 1.
- batch_size : Size of the batch.
- classes : Dictionary with the classes as the keys and their pixel value as the values.
- matrice_file : Matrice file containing the patches, the mean and std. If not declared, the parameter output_file in the section saving is used.
- experiment_folder : Output folder for the training. A folder *model* will be created in this folder.
- patch_size : Size of the patches. If not defined, the parameter patch_size in the general section is used.
- data_augmentation : if true, process data augmentation. You can the nadd some parameters of the data augmentation such as rotation_range, width_shift_range, height_shift_rangerescale, zoom_range, horizontal_flip, vertical_flip. For more information you can visit https://keras.io/preprocessing/image/.

#### Segmentation
- filenames  : Array of globs matching the input images.
- mask_pattern : Maskname pattern (something like _mask.png for example)
- matrice_file  : Matrice file containing the patches, the mean and std. If not declared, the parameter output_file in the section saving is used.  
- output_pattern : Suffix added to the file name for the prediction image. For example, if you are working on the file toto.png and put "_pred.png" here, the output file will be : toto_pred.png.
- experiment_folder : Input folder. If not declared, the parameter experiment_folder in the general section is used. The folder prediction in this folder will be used.
- model_name : Matrice file generated by the training and used for the prediction.
- load_level : Parameter for SVS file. Define the loading level for the image. If not defined, the parameter load_level in the general section is used.
- patch_size : Size of the patches. If not defined, the parameter patch_size in the general section is used.
- stride : Stride use for the segmentation (The image is splitted into patches for the segmentation, that's why we need a stride).
- use_training_norm: If true, use the mean and std of the training, else use the mean and std of the image.
- show_prediction : If true, the images will be shown in the notebook.
- nb_classes : Number of classes to recognize.
- color_channels : The number of chnnaels in the image (3 for RGB for example). If not defined, the parameter color_channels of the general section is used.

#### Evaluation
- experiment_folder : Input folder. If not declared, the parameter experiment_folder in the general section is used. The folder prediction in this folder will be used. A folder *predictions* will created in this folder.
- images : Array of globs matching the original images.
- transparency : Alpha channel value for the annoatation on the output images.
- display : If true, the images will be shown in the notebook.
- load_level : If not declared, the parameter load_level in the general section is used.

## Other information

When you are done with a step, you should shutdown it in order to free the memory (especially with training and segmentation). 
If you want to use another parameter from the configuration file, you can use ${[section]:[parameter]}. For example, ${general:patch_size} will use the patch_size parameter from the general section.