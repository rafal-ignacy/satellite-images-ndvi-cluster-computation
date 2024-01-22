from PIL import Image
from itertools import product
import os

def merge(dir_in, dir_out, tile_size):
    files = os.listdir(dir_in)
    
    positions = [(int(fname.split('_')[1]), int(fname.split('_')[2].split('.')[0])) for fname in files]

    size_x = sum(1 for position in positions if position[0] == 0)*tile_size
    size_y = sum(1 for position in positions if position[1] == 0)*tile_size
    
    # merged_img = Image.new('RGB', (size_x, size_y), (255,255,255)) #for PNG and JPG images
    merged_img = Image.new('I', (size_x , size_y))

    for filename, position in zip(files, positions):
        img = Image.open(os.path.join(dir_in, filename))
        position = (position[1], position[0])
        merged_img.paste(img, position)

    merged_path = os.path.join(dir_out, 'merged_B5_image.TIF')
    merged_img.save(merged_path)

def split(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)

tile_size = 1000
filename = 'LC08_L1TP_129034_20231228_20240108_02_T1_B4.TIF'
dir_in = 'sat_images'
dir_out = 'out'
after_merge = 'after_merge'
split(filename, dir_in, dir_out,tile_size)
merge(dir_out, after_merge, tile_size)