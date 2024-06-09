import cv2
import glob
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def is_black_patch(patch):
    return np.all(patch == 0)

def calculate_black_area_ratio(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black_area = sum(cv2.contourArea(contour) for contour in contours)
    total_area = patch.shape[0] * patch.shape[1]
    return black_area / total_area

def save_patches(image_path, patch_size, output_dir, make_black_image, make_black_zone):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    base_name = os.path.basename(image_path).split('.')[0]

    patch_index = 0
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = img[y:y+patch_size, x:x+patch_size]
            if not make_black_zone:
                black_area_ratio = calculate_black_area_ratio(patch)
                if black_area_ratio > 0.05:
                    continue
            if any([make_black_image, not is_black_patch(patch)]):
                patch_name = f'{base_name}_patch_{patch_index}.png'
                cv2.imwrite(os.path.join(output_dir, patch_name), patch)
                patch_index += 1

def process_image(image_path):
    save_patches(image_path, patch_size, output_dir, make_black_image, make_black_zone)

input_dir = r'C:\Users\user\Desktop\SAR_Image\Dataset\SPACENET\train\transform\SAR-Intensity_2-998scaling_stretch_png'
output_dir = r'C:\Users\user\Desktop\SAR_Image\Dataset\SPACENET\train\transform\SAR-Intensity_2-998scaling_stretch_png_256'
patch_size = 256
make_black_image = False
make_black_zone = False

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_paths = glob.glob(input_dir + '/*.png')

cpu_count = os.cpu_count()
with ThreadPoolExecutor(max_workers=cpu_count) as executor:
    list(tqdm(executor.map(process_image, image_paths), total=len(image_paths)))