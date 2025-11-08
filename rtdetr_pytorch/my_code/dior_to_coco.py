from pathlib import Path

# 复制图片

## 定义路径

################################################################
dior_raw_path = "/root/autodl-tmp/DIOR/raw"
###############################################################
dior_set_main_path = Path(dior_raw_path) / "ImageSets/Main"
print(dior_set_main_path)
dior_set_train_path = Path(dior_set_main_path) / "train.txt"
dior_set_val_path = Path(dior_set_main_path) / "val.txt"
dior_set_test_path = Path(dior_set_main_path) / "test.txt"
dior_train_img_path = Path(dior_raw_path) / "JPEGImages-trainval"
dior_val_img_path = Path(dior_raw_path) / "JPEGImages-trainval"
dior_test_img_path = Path(dior_raw_path) / "JPEGImages-test"

################################################################
dior_coco_path = "/root/autodl-tmp/DIOR_coco"
################################################################
dior_coco_anno_path = Path(dior_coco_path) / "annotations"
dior_coco_train_img_path = Path(dior_coco_path) / "train"
dior_coco_val_img_path = Path(dior_coco_path) / "val"
dior_coco_test_img_path = Path(dior_coco_path) / "test"
print(dior_coco_train_img_path)

## 获取集合信息

set_train_all = []

def get_set_all(set_text_path): 
    set_all = []
    with open(set_text_path, 'r', encoding='utf-8') as file:
        set_all = file.readlines()
    for i in range(0, len(set_all)):
        set_all[i] = set_all[i].strip()
    return set_all 

set_train_all = get_set_all(dior_set_train_path)
print(f'train total: {len(set_train_all)}')
# print(set_train_all)

set_val_all = get_set_all(dior_set_val_path)
print(f'val total: {len(set_val_all)}')
# print(set_val_all)

set_test_all = get_set_all(dior_set_test_path)
print(f'test total: {len(set_test_all)}')
# print(set_test_all)

## 复制图片
import shutil
from pathlib import Path
import os

"""
检查是否是空目录，是空目录再执行复制操作，不是空目录则报错退出
"""
def copy_img_to_coco_dir(set_all, dior_img_dir, coco_img_dir):
    try:
        if len(os.listdir(coco_img_dir)) != 0:
            print("Target directory cannot contain any files or subdirectories.")
            return 
        img_name_all = []
        for one_ in set_all:
            img_name_all.append(one_ + '.jpg')
        # print(img_name_all)
        print(f'from:{dior_img_dir}\ncopy to:{coco_img_dir}\n')
        for img_name in img_name_all:        
            # 确保源文件存在
            src_path = Path(dior_img_dir) / img_name
            dst_path = Path(coco_img_dir)
            # dst_path = coco_img_dir
            # print(str(src_path))
            if not os.path.isfile(src_path):
                print(f"错误: 源文件 '{src_path}' 不存在或不是文件。")
                return 
            # 使用 copy2 复制文件，并保留元数据
            shutil.copy2(src_path, dst_path)
            # print(f"文件已成功从 '{src}' 复制到 '{dst}'。")
        print("done")
    except PermissionError:
        print(f"错误: 权限不足，无法读取源文件或写入目标位置。")
        return False
    except FileNotFoundError:
        print(f"错误: 找不到源文件或目标目录。")
        return False
    except Exception as e:
        print(f"复制文件时发生未知错误: {e}")
        return False


copy_img_to_coco_dir(set_train_all, dior_train_img_path, dior_coco_train_img_path)
copy_img_to_coco_dir(set_val_all, dior_val_img_path, dior_coco_val_img_path)
copy_img_to_coco_dir(set_test_all, dior_test_img_path, dior_coco_test_img_path)
# shutil.copy2("/root/autodl-tmp/DIOR/raw/JPEGImages-trainval/00001.jpg", "/root/autodl-tmp/DIOR_coco/train")


# 处理标注信息
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path

categories = [
    "golffield", "vehicle", "Expressway_toll_station",
    "trainstation", "chimney", "storagetank", "ship", 
    "harbor", "airplane", "tenniscourt", "groundtrackfield", 
    "dam", "basketballcourt", "Expressway_Service_area", 
    "stadium", "airport", "baseballfield", "bridge", 
    "windmill", "overpass"
]

# 创建类别ID映射
category_dict = {name: idx + 1 for idx, name in enumerate(categories)}
# print(category_dict)

for idx, name in enumerate(categories):
    print(f'{idx+1} : \'{name}\',')

def voc_to_coco(set_all, dior_raw_path, output_path):
    """
    将VOC数据集转换为COCO格式
    
    Args:
        set_all: 某个集合的图片标号信息
        dior_root: DIOR 数据集根目录路径
        output_path: 输出COCO JSON文件路径
    """
    
    # VOC类别映射（按字母顺序排序）
    categories = [
        "golffield", "vehicle", "Expressway_toll_station",
        "trainstation", "chimney", "storagetank", "ship", 
        "harbor", "airplane", "tenniscourt", "groundtrackfield", 
        "dam", "basketballcourt", "Expressway_Service_area", 
        "stadium", "airport", "baseballfield", "bridge", 
        "windmill", "overpass"
    ]
    
    # 创建类别ID映射
    category_dict = {name: idx + 1 for idx, name in enumerate(categories)}
    
    # 初始化COCO结构
    coco_data = {
        "info": {  "year": "2025",
                    "version": "1.0",
                    "description": "DIOR Dataset in COCO format",
                    "contributor": "jieyi",
                    "url": "none",
                    "date_created": "2025-11-9"},
        "licenses": [
                    {
                    "id": 1,
                    "name": "Attribution-NonCommercial-ShareAlike License",
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                    }],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 添加类别信息
    for idx, name in enumerate(categories):
        coco_data["categories"].append({
            "id": idx + 1,
            "name": name,
            "supercategory": "object"
        })
    
    image_id = 1
    annotation_id = 1
    
    # VOC标注文件目录
    annotations_dir = Path(dior_raw_path) / "Annotations/Horizontal Bounding Boxes"
    
    xml_file_all = []
    for one_ in set_all:
        xml_file_all.append(one_ + '.xml')
    # print(xml_file_all)
    for xml_file in xml_file_all:
        xml_file_path = annotations_dir / xml_file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # 获取图像信息
        filename = root.find("filename").text
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        
        # 添加图像信息
        coco_data["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })
        
        # 处理每个对象
        for obj in root.findall("object"):
            name = obj.find("name").text.lower().strip()
            
            # 跳过未知类别
            if name not in category_dict:
                continue
            
            # 获取边界框
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)
            
            # 转换为COCO格式 [x, y, width, height]
            x = xmin
            y = ymin
            width_bbox = xmax - xmin
            height_bbox = ymax - ymin
            # 防止出现标注框不存在的问题
            if width_bbox <= 0:
                print(f"宽度非正数，已修正: {width_bbox} -> 1.0")
                width_bbox = 1.0  # 或者跳过这个标注
            if height_bbox <= 0:
                print(f"高度非正数，已修正: {height_bbox} -> 1.0")
                height_bbox = 1.0 # 或者跳过这个标注
            
            # 计算面积
            area = width_bbox * height_bbox
            
            # 添加标注
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_dict[name],
                "bbox": [x, y, width_bbox, height_bbox],
                "segmentation": [],  # DIOR没有分割信息
                "area": area,
                "iscrowd": 0
            })
            
            annotation_id += 1
        
        image_id += 1
    
    # 保存到JSON文件
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"转换完成！共处理 {image_id-1} 张图像，{annotation_id-1} 个标注")
    print(f"输出文件: {output_path}")
    

dior_coco_anno_train_path = Path(dior_coco_anno_path) / "train.json"
voc_to_coco(set_train_all, dior_raw_path, dior_coco_anno_train_path)

dior_coco_anno_val_path = Path(dior_coco_anno_path) / "val.json"
voc_to_coco(set_val_all, dior_raw_path, dior_coco_anno_val_path)

dior_coco_anno_test_path = Path(dior_coco_anno_path) / "test.json"
voc_to_coco(set_test_all, dior_raw_path, dior_coco_anno_test_path)