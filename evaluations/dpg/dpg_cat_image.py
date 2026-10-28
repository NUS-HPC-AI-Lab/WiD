import cv2
import numpy as np
import os
import pathlib
import argparse

def group_images(path_list):
    sorted(path_list)
    class_id_dict = {}
    for path in path_list:
        class_id = str(path.name).split('_')[0]
        if class_id not in class_id_dict:
            class_id_dict[class_id] = []
        class_id_dict[class_id].append(path)
    return class_id_dict

def cat_images(path_list):
    imgs = []
    for path in path_list:
        img = cv2.imread(str(path))
        os.remove(path)
        imgs.append(img)
    row_cat_images = []
    row_length = int(len(imgs)**0.5)
    for i in range(len(imgs)//row_length):
        row_cat_images.append(np.concatenate(imgs[i*row_length:(i+1)*row_length], axis=1))
    cat_image = np.concatenate(row_cat_images, axis=0)
    return cat_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default=None)
    
    args = parser.parse_args()
    src_dir = args.src_dir
    path_list = list(pathlib.Path(src_dir).glob('*.png'))
    class_id_dict = group_images(path_list)
    for class_id, path_list in class_id_dict.items():
        cat_image = cat_images(path_list)
        cat_path = os.path.join(src_dir, f'{class_id}.jpg')
        # cat_path = "cat_{}.png".format(class_id)
        cv2.imwrite(cat_path, cat_image)