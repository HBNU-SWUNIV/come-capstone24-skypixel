from concurrent.futures import ThreadPoolExecutor
import glob
from osgeo import gdal
import numpy as np
from PIL import Image
import os

from tqdm import tqdm

def find_min_max_values(input_files, channels, low_scaling, high_scaling):
    min_val = np.inf
    max_val = -np.inf
    for file in tqdm(input_files, total=len(input_files), desc="Finding min and max values"):
        dataset = gdal.Open(file)
        all_data = []
        for ch in channels:
            band = dataset.GetRasterBand(ch)
            band_data = band.ReadAsArray()
            all_data.append(band_data.flatten())
        all_data = np.concatenate(all_data)
        all_data_sorted = np.sort(all_data)
        low_index = int(len(all_data_sorted) * (low_scaling / 100.0))
        high_index = int(len(all_data_sorted) * (high_scaling / 100.0))
        min_val = min(all_data_sorted[low_index], min_val)
        max_val = max(all_data_sorted[high_index], max_val)
            
    return min_val, max_val

def convert_tif_to(input_file, output_file, channels, min_val=None, max_val=None, stretch=False):
    dataset = gdal.Open(input_file)
    if len(channels) == 1:
        channel_data = dataset.GetRasterBand(channels[0]).ReadAsArray()
        data = channel_data
    else:
        data = np.dstack([dataset.GetRasterBand(ch).ReadAsArray() for ch in channels])

    if stretch and min_val is not None and max_val is not None:
        # Stretching
        data = np.clip(data, min_val, max_val)  # 데이터를 최소값과 최대값 사이로 제한
        data = (data - min_val) / (max_val - min_val) * 255  # 정규화 및 스케일 조정
        data = data.astype('uint8')  # uint8 타입으로 변환
    else:
        # 데이터 타입 변환 (부동 소수점에서 8비트 정수로)
        data = np.clip(data, 0, 255)  # 데이터를 0과 255 사이로 제한
        data = data.astype('uint8')  # uint8 타입으로 변환

    image = Image.fromarray(data)
    image.save(output_file)

    aux_file = output_file + '.aux.xml'
    if os.path.exists(aux_file):
        os.remove(aux_file)

def process_files(input_files, output_dir, channels, stretch, min_val, max_val):
    for file in input_files:
        base_name, extension = os.path.splitext(os.path.basename(file))
        output_file = os.path.join(output_dir, base_name + '.png')
        convert_tif_to(file, output_file, channels, min_val, max_val, stretch)

# 파일 경로 정의
input_dir = r'C:\Users\user\Desktop\SAR_Image\Dataset\SPACENET\train\original\SAR-Intensity'
output_dir = r'C:\Users\user\Desktop\SAR_Image\Dataset\SPACENET\train\transform\SAR-Intensity_2-998scaling_stretch_png'

channels = [1]
stretch = True
low_scaling = 0.2
high_scaling = 99.8

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 입력 디렉토리에서 모든 파일을 찾습니다.
input_files = glob.glob(input_dir + '/*.tif')

if stretch:
    min_val, max_val = find_min_max_values(input_files, channels, low_scaling, high_scaling)
    print("min_val : ", min_val)
    print("max_val : ", max_val)
else:
    min_val, max_val = None, None

# ThreadPoolExecutor를 사용하여 병렬 처리
cpu_count = os.cpu_count()
with ThreadPoolExecutor(max_workers=cpu_count) as executor:
    list(tqdm(executor.map(lambda file: process_files([file], output_dir, channels, stretch, min_val, max_val), input_files), total=len(input_files), desc="Converting TIF to PNG"))
