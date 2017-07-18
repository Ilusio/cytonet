import os
import glob
from skimage import morphology

def mkdirs(folder, permission):
    if not os.path.exists(folder):
        try:
            original_umask = os.umask(0)
            os.makedirs(folder,permission)
        finally:
            os.umask(original_umask)
            
def extend_glob(filenames):
    files= []
    for filename in filenames:
        files.extend(glob.glob(filename))
    return files

def file_suffix(file, suffix):
    name = os.path.splitext(os.path.basename(file))[0]
    new_name = os.path.dirname(file)+"/"+name+suffix
    return new_name

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img