import os
import numpy as np
from osgeo import gdal
from skimage.color import rgb2hsv, hsv2rgb
from skimage.exposure import match_histograms
import cv2


def normalize_interpolation_matrix(interpolation_matrix):
    """
    归一化插值矩阵
    :param interpolation_matrix: 插值矩阵
    :return: 归一化后的插值矩阵
    """
    min_val = np.min(interpolation_matrix)
    max_val = np.max(interpolation_matrix)
    normalized_matrix = (interpolation_matrix - min_val) / (max_val - min_val)
    return normalized_matrix

def smooth_interpolation_matrix(interpolation_matrix, kernel_size=3):
    """
    对插值矩阵进行平滑处理
    :param interpolation_matrix: 插值矩阵
    :param kernel_size: 平滑滤波器的核大小
    :return: 平滑后的插值矩阵
    """
    smoothed_matrix = cv2.GaussianBlur(interpolation_matrix, (kernel_size, kernel_size), 0)
    return smoothed_matrix

def weighted_fusion(hsi_image, interpolation_matrix, alpha=0.5):
    """
    加权融合
    :param hsi_image: 高光谱图像 (H x W x C)
    :param interpolation_matrix: 插值矩阵 (H x W)
    :param alpha: 权重参数
    :return: 融合后的图像 (H x W x C)
    """
    fused_image = hsi_image.copy().astype(np.float32)
    for i in range(hsi_image.shape[2]):
        fused_image[:, :, i] = (1 - alpha) * hsi_image[:, :, i] + alpha * interpolation_matrix
    fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)
    return fused_image
def read_tif(file_path):
    try:
        dataset = gdal.Open(file_path)
        if dataset is None:
            raise ValueError(f"Failed to open {file_path}")
        bands = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
        img = np.stack(bands, axis=-1)
        return img
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# def spectral_correction(original_msi, fused_msi):
#     # 计算光谱特征
#     original_mean = np.mean(original_msi, axis=(0, 1))
#     fused_mean = np.mean(fused_msi, axis=(0, 1))
#
#     # 计算光谱矫正系数
#     correction_factors = original_mean / (fused_mean + 1e-6)
#
#     # 应用光谱矫正
#     corrected_fused_msi = fused_msi * correction_factors
#
#     return np.clip(corrected_fused_msi, 0, 255).astype(np.uint8)
# def ihs_fusion(high_res_band, low_res_image):
#     # 将 3 波段图像从 BGR 转换到 HSV 空间
#     low_res_hsv = cv2.cvtColor(low_res_image, cv2.COLOR_BGR2RGB)
#
#     # 计算高分辨率波段的强度分量
#     high_res_intensity = high_res_band / 255.0  # 归一化到 [0, 1]
#
#     # 进行直方图匹配
#     matched_intensity = match_histograms(high_res_band, low_res_hsv[:, :, 2])
#
#     # 替换低分辨率图像的强度分量
#     fused_hsv = low_res_hsv.copy()
#     fused_hsv[:, :, 2] = matched_intensity
#
#     # 将 HSV 图像转换回 BGR 空间
#     fused_bgr = cv2.cvtColor(fused_hsv.astype(np.uint8), cv2.COLOR_RGB2BGR)
#
#     return fused_bgr
def spectral_correction(original_msi, fused_msi):
    # 计算光谱特征
    original_mean = np.mean(original_msi, axis=(0, 1))
    fused_mean = np.mean(fused_msi, axis=(0, 1))

    # 计算光谱矫正系数
    correction_factors = original_mean / (fused_mean + 1e-6)

    # 应用光谱矫正
    corrected_fused_msi = fused_msi * correction_factors

    return np.clip(corrected_fused_msi, 0, 255).astype(np.uint8)


def ihs_fusion(high_res_band, low_res_image):
    # 将 3 波段图像从 BGR 转换到 HSV 空间
    I = np.mean(low_res_image,axis=-1)

    # low_res_hsv = cv2.cvtColor(low_res_image, cv2.COLOR_BGR2RGB)

    # 计算高分辨率波段的强度分量
    # high_res_intensity = high_res_band / 255.0  # 归一化到 [0, 1]

    # 进行直方图匹配
    matched_intensity = match_histograms( high_res_band,I)
    D = matched_intensity -I
    # normalized_matrix = normalize_interpolation_matrix(D)

    # 替换低分辨率图像的强度分量
    fused_hsv = low_res_image + D[...,np.newaxis]
    # fused_hsv[:, :, 2] = matched_intensity
    corrected_fused_msi = spectral_correction(low_res_image, fused_hsv)
    # 将 HSV 图像转换回 BGR 空间
    fused_bgr = cv2.cvtColor(corrected_fused_msi .astype(np.uint8), cv2.COLOR_RGB2BGR)
    fused_image = weighted_fusion(low_res_image, D, alpha=0.3)

    return fused_bgr
    # return fused_hsv

def process_folders(high_res_folder, low_res_folder, output_folder1, output_folder2):
    if not os.path.exists(output_folder1):
        os.makedirs(output_folder1)
    if not os.path.exists(output_folder2):
        os.makedirs(output_folder2)

    skipped_files = []
    unreadable_files = []

    for high_res_filename in os.listdir(high_res_folder):
        if high_res_filename.endswith('.tif'):
            high_res_file_path = os.path.join(high_res_folder, high_res_filename)
            low_res_file_path = os.path.join(low_res_folder, high_res_filename)

            if os.path.exists(low_res_file_path):
                high_res_image = read_tif(high_res_file_path)
                low_res_image = read_tif(low_res_file_path)

                if high_res_image is None or low_res_image is None:
                    unreadable_files.append(high_res_filename)
                    continue

                # 确保图像寸一致

                    # 检查图像尺寸一致性
                if high_res_image.shape[:2] != low_res_image.shape[:2]:
                    print(f"Skipping {high_res_filename}: Image sizes do not match")
                    skipped_files.append(high_res_filename)
                    continue

                # 提取 2 波段图像的两个波段
                band1 = high_res_image[:, :, 0]
                band2 = high_res_image[:, :, 1]

                fused_band1 = ihs_fusion(band1, low_res_image)  # 传入RGB图像
                fused_band2 = ihs_fusion(band2, low_res_image)

                # 保存融合结果
                output_file_path1 = os.path.join(output_folder1, high_res_filename)
                output_file_path2 = os.path.join(output_folder2, high_res_filename)

                cv2.imwrite(output_file_path1, fused_band1)
                cv2.imwrite(output_file_path2, fused_band2)

                print(f"Processed and saved {high_res_filename}")

    if skipped_files:
        print("Skipped files due to size mismatch:")
        for file in skipped_files:
            print(file)

    if unreadable_files:
        print("Unreadable files:")
        for file in unreadable_files:
            print(file)

# 指定文件夹路径
high_res_folder = 'dataset_color/dataset3/SAR'
low_res_folder = 'dataset_color/dataset3/OPT'
output_folder1 = 'output_IHS_data3/VV'
output_folder2 = 'output_IHS_data3/VH'

# 处理文件夹
process_folders(high_res_folder, low_res_folder, output_folder1, output_folder2)
