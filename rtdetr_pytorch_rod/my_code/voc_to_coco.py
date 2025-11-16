import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path

# def diorvoc_to_coco( dior_root, output_path ):
#     categories = [
#         "golffield", "vehicle", "Expressway_toll_station",
#         "trainstation", "chimney", "storagetank", "ship", 
#         "harbor", "airplane", "tenniscourt", "groundtrackfield", 
#         "dam", "basketballcourt", "Expressway_Service_area", 
#         "stadium", "airport", "baseballfield", "bridge", 
#         "windmill", "overpass"
#     ]
    
#     # 创建类别ID映射
#     category_dict = {name: idx + 1 for idx, name in enumerate(categories)}

def voc_to_coco(voc_root, output_path):
    """
    将VOC数据集转换为COCO格式
    
    Args:
        voc_root: VOC数据集根目录路径
        output_path: 输出COCO JSON文件路径
    """
    
    # VOC类别映射（按字母顺序排序）
    categories = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    # 创建类别ID映射
    category_dict = {name: idx + 1 for idx, name in enumerate(categories)}
    
    # 初始化COCO结构
    coco_data = {
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
    annotations_dir = Path(voc_root) / "Annotations"
    
    for xml_file in annotations_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
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
            
            # 计算面积
            area = width_bbox * height_bbox
            
            # 添加标注
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_dict[name],
                "bbox": [x, y, width_bbox, height_bbox],
                "segmentation": [],  # VOC没有分割信息
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

# 使用示例
if __name__ == "__main__":
    voc_root = "/path/to/VOCdevkit/VOC2007"  # 修改为你的VOC路径
    output_path = "instances_voc2007.json"
    
    voc_to_coco(voc_root, output_path)