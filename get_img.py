import os, glob
import shutil

img_path = '/home/huyvd/Documents/works/crnn-pytorch-master/data/test'
save_path = '/home/huyvd/Documents/works/test_crnn'

list_dir = glob.glob(os.path.join(img_path, '*'))
for i, dir in enumerate(list_dir):
    if i % 25 == 0:
        shutil.copy(dir, save_path)
