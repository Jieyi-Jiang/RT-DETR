import random
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO

def get_random_color():
    """生成随机颜色"""
    return tuple(random.randint(0, 255) for _ in range(3))

def get_high_saturated_color(color):
    """提高颜色的饱和度"""
    return tuple(min(255, int(c * 1.5)) for c in color)

def visualize_and_save_coco_annotation(coco, img_dir, img_number, output_path):
    """
    根据图片编号可视化COCO标注，并将结果保存为图片文件。
    
    :param coco: COCO实例
    :param img_dir: 图像所在目录
    :param img_number: 图片编号
    :param output_path: 输出图片文件路径
    """
    selected_img_id = img_number - 5863 + 1
    
    # 获取图像信息
    img_info = coco.loadImgs(selected_img_id)[0]
    img_path = f"{img_dir}/{img_info['file_name']}"
    
    # 加载图像
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    ann_ids = coco.getAnnIds(imgIds=img_info['id'])
    annotations = coco.loadAnns(ann_ids)

    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    cat_to_color = {cat['id']: get_high_saturated_color(get_random_color()) for cat in cats}

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for annotation in annotations:
        bbox = annotation['bbox']
        category_id = annotation['category_id']
        category_name = coco.loadCats([category_id])[0]['name']
        
        color = cat_to_color[category_id]

        if bbox:
            x, y, w, h = bbox
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            
            # 同时显示类别ID和名称
            label_text = f'[{category_id}] {category_name}'
            text_bbox = draw.textbbox((x, y), label_text, font=font)
            text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.rectangle([x, y - text_h - 5, x + text_w + 5, y], fill=color)
            draw.text((x + 3, y - text_h - 2), label_text, fill='white', font=font)

    image.save(output_path, quality=95)  # 直接保存PIL图像对象
    print(f"Processed image saved to {output_path}")

def main(img_number):
    annotation_file = "/root/autodl-tmp/DIOR_coco/annotations/val.json"
    img_dir = "/root/autodl-tmp/DIOR_coco/val"

    coco = COCO(annotation_file)

    output_path = "./dior_img_gt.jpg" 

    visualize_and_save_coco_annotation(coco, img_dir, img_number, output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument('img_number', type=int, help='The image number for processing.')
    args = parser.parse_args()
    main(args.img_number)