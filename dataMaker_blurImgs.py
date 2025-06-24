import os
import shutil
import cv2
import numpy as np
from VQ.common_func import parse_label

def random_gaussian_blur(image, kernel_size=(5, 7, 9), sigma_range=(0.5, 3.0)):
    """
    对输入的图像应用随机高斯模糊。
    :param image: 输入的图像数据
    :return: 经过随机高斯模糊处理后的图像
    """
    # 随机选择高斯核的大小，核大小必须是正奇数
    kernel_size = np.random.choice(kernel_size)
    # 随机选择标准差，这里设置范围在 0.5 到 3.0 之间
    sigma = np.random.uniform(*sigma_range)
    # 应用高斯模糊
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return blurred_image


def is_zero(file_name: str):
    if file_name.split('-')[0] == '0':
        return True
    else:
        return False


def process_png_files(root_dir, skip_function=None):
    """
    遍历指定目录及其子目录，处理所有 PNG 图片文件，应用随机高斯模糊并覆盖原始图片。
    :param skip_function: 给文件名，返回 true or false, true 则跳过该文件
    :param root_dir: 要遍历的根目录路径
    :return: 无
    注意事项:
        - 该函数会直接覆盖原始的 PNG 文件，请确保在运行前备份重要数据。
        - 请确保有足够的权限对文件进行读写操作。
    """
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if skip_function and skip_function(file):
                continue
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                try:
                    # 读取图片
                    image = cv2.imread(file_path)
                    if image is not None:
                        # 应用随机高斯模糊
                        blurred_image = random_gaussian_blur(image)
                        # 保存模糊后的图片，覆盖原始图片
                        cv2.imwrite(file_path, blurred_image)
                        print(f"Processed and replaced: {file_path}")
                    else:
                        print(f"Failed to read image: {file_path}")
                except Exception as e:
                    print(f"Error processing image {file_path}: {e}")

def copy_subdirectories_n_times(source_dir, target_dir, n):
    """
    将源目录下的所有一级子目录复制 n 次到目标目录。
    :param source_dir: 源目录的路径
    :param target_dir: 目标目录的路径
    :param n: 复制的次数
    :return: 无
    注意事项:
        - 若目标目录下的目标子目录已存在，会跳过复制操作。
        - 请确保有足够的权限在目标目录下创建新的子目录和复制文件。
    """
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return
    # 检查目标目录是否存在，不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # 获取源目录下的所有一级子目录
    subdirectories = [name for name in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, name))]
    for sub_dir in subdirectories:
        sub_dir_path = os.path.join(source_dir, sub_dir)
        for i in range(1, n + 1):
            new_sub_dir_name = f"{sub_dir}-{i}"
            new_sub_dir_path = os.path.join(target_dir, new_sub_dir_name)
            try:
                # 复制子目录及其内容
                shutil.copytree(sub_dir_path, new_sub_dir_path)
                print(f"Successfully copied {sub_dir_path} to {new_sub_dir_path}")
            except FileExistsError:
                print(f"Target directory {new_sub_dir_path} already exists, skipping copy.")
            except Exception as e:
                print(f"Error copying {sub_dir_path} to {new_sub_dir_path}: {e}")

def copy_files_n_times(source_dir, target_dir, n):
    """
    将源目录下的所有文件复制 n 次到目标目录。
    :param source_dir: 源目录的路径
    :param target_dir: 目标目录的路径
    :param n: 复制的次数
    :return: 无
    注意事项:
        - 该函数仅处理源目录下的一级文件，不会递归处理子目录中的文件。
        - 请确保有足够的权限在目标目录下创建新文件和复制文件。
    """
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return

    # 检查目标目录是否存在，不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 获取源目录下的所有文件
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    for file in files:
        file_path = os.path.join(source_dir, file)
        file_name, file_ext = os.path.splitext(file)
        for i in range(1, n + 1):
            new_file_name = f"{file_name}-{i}{file_ext}"
            new_file_path = os.path.join(target_dir, new_file_name)
            try:
                # 复制文件
                shutil.copy2(file_path, new_file_path)
                print(f"Successfully copied {file_path} to {new_file_path}")
            except Exception as e:
                print(f"Error copying {file_path} to {new_file_path}: {e}")


if __name__ == "__main__":
    '''
        复制并虚化 (0,20)-FixedPos-oneStyle
    '''
    # source_oneStyle_directory_path = './dataset/(0,20)-FixedPos-oneStyle'
    # target_oneStyle_directory_path = './dataset/blur-(0,20)-FixedPos-oneStyle'
    # copy_files_n_times(source_oneStyle_directory_path, target_oneStyle_directory_path, 16)
    # # 虚化
    # process_png_files(target_oneStyle_directory_path, skip_function=is_zero)

    '''
        复制并虚化 single_style_pairs(0,20)_tripleSet
    '''
    # source_tripleSet_directory_path = './dataset/single_style_pairs(0,20)_tripleSet'
    # target_tripleSet_directory_path = './dataset/blur-single_style_pairs(0,20)_tripleSet'
    # os.makedirs(target_tripleSet_directory_path, exist_ok=True)
    # copy_subdirectories_n_times(f'{source_tripleSet_directory_path}/train',
    #                             f'{target_tripleSet_directory_path}/train', 16)
    # copy_subdirectories_n_times(f'{source_tripleSet_directory_path}/test_1',
    #                             f'{target_tripleSet_directory_path}/test_1', 16)
    # copy_subdirectories_n_times(f'{source_tripleSet_directory_path}/test_2',
    #                             f'{target_tripleSet_directory_path}/test_2', 16)
    # # 虚化
    # process_png_files(target_tripleSet_directory_path, skip_function=lambda x: parse_label(x) == 0)

    '''
        复制并虚化 single_style_pairs(0,20)_tripleSet_trainAll
    '''
    # source_tripleSet_directory_path = './dataset/single_style_pairs(0,20)_tripleSet_trainAll'
    # target_tripleSet_directory_path = './dataset/blur-single_style_pairs(0,20)_tripleSet_trainAll'
    # os.makedirs(target_tripleSet_directory_path, exist_ok=True)
    # copy_subdirectories_n_times(f'{source_tripleSet_directory_path}/train',
    #                             f'{target_tripleSet_directory_path}/train', 16)
    # copy_subdirectories_n_times(f'{source_tripleSet_directory_path}/test_1',
    #                             f'{target_tripleSet_directory_path}/test_1', 16)
    # copy_subdirectories_n_times(f'{source_tripleSet_directory_path}/test_2',
    #                             f'{target_tripleSet_directory_path}/test_2', 16)
    # # 虚化
    # process_png_files(target_tripleSet_directory_path, skip_function=lambda x: parse_label(x) == 0)

    '''
        读取测试图片
    '''
    image = cv2.imread('./z_test/1-circle-blue.png')
    if image is not None:
        # 应用随机高斯模糊
        # blurred_image = random_gaussian_blur(image)
        blurred_image = cv2.GaussianBlur(image, (19, 19), 19)  # 使用固定的核大小和标准差
        # 显示原始图片和模糊后的图片
        cv2.imshow('Original Image', image)
        cv2.imshow('Blurred Image', blurred_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("无法读取图片，请检查图片路径。")