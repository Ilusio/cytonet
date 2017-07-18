import os
import glob

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