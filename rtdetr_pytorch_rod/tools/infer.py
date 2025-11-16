import torch
import torch.nn as nn 
import torchvision.transforms as T
from torch.cuda.amp import autocast
import numpy as np 
from PIL import Image, ImageDraw, ImageFont
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
import numpy as np

def postprocess(labels, boxes, scores, iou_threshold=0.55):
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        return iou
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    used_indices = set()
    for i in range(len(boxes)):
        if i in used_indices:
            continue
        current_box = boxes[i]
        current_label = labels[i]
        current_score = scores[i]
        boxes_to_merge = [current_box]
        scores_to_merge = [current_score]
        used_indices.add(i)
        for j in range(i + 1, len(boxes)):
            if j in used_indices:
                continue
            if labels[j] != current_label:
                continue  
            other_box = boxes[j]
            iou = calculate_iou(current_box, other_box)
            if iou >= iou_threshold:
                boxes_to_merge.append(other_box.tolist())  
                scores_to_merge.append(scores[j])
                used_indices.add(j)
        xs = np.concatenate([[box[0], box[2]] for box in boxes_to_merge])
        ys = np.concatenate([[box[1], box[3]] for box in boxes_to_merge])
        merged_box = [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]
        merged_score = max(scores_to_merge)
        merged_boxes.append(merged_box)
        merged_labels.append(current_label)
        merged_scores.append(merged_score)
    return [np.array(merged_labels)], [np.array(merged_boxes)], [np.array(merged_scores)]
def slice_image(image, slice_height, slice_width, overlap_ratio):
    img_width, img_height = image.size
    
    slices = []
    coordinates = []
    step_x = int(slice_width * (1 - overlap_ratio))
    step_y = int(slice_height * (1 - overlap_ratio))
    
    for y in range(0, img_height, step_y):
        for x in range(0, img_width, step_x):
            box = (x, y, min(x + slice_width, img_width), min(y + slice_height, img_height))
            slice_img = image.crop(box)
            slices.append(slice_img)
            coordinates.append((x, y))
    return slices, coordinates
def merge_predictions(predictions, slice_coordinates, orig_image_size, slice_width, slice_height, threshold=0.30):
    merged_labels = []
    merged_boxes = []
    merged_scores = []
    orig_height, orig_width = orig_image_size
    for i, (label, boxes, scores) in enumerate(predictions):
        x_shift, y_shift = slice_coordinates[i]
        scores = np.array(scores).reshape(-1)
        valid_indices = scores > threshold
        valid_labels = np.array(label).reshape(-1)[valid_indices]
        valid_boxes = np.array(boxes).reshape(-1, 4)[valid_indices]
        valid_scores = scores[valid_indices]
        for j, box in enumerate(valid_boxes):
            box[0] = np.clip(box[0] + x_shift, 0, orig_width)  
            box[1] = np.clip(box[1] + y_shift, 0, orig_height)
            box[2] = np.clip(box[2] + x_shift, 0, orig_width)  
            box[3] = np.clip(box[3] + y_shift, 0, orig_height) 
            valid_boxes[j] = box
        merged_labels.extend(valid_labels)
        merged_boxes.extend(valid_boxes)
        merged_scores.extend(valid_scores)
    return np.array(merged_labels), np.array(merged_boxes), np.array(merged_scores)
# def draw(images, labels, boxes, scores, thrh = 0.6, path = ""):
#     for i, im in enumerate(images):
#         draw = ImageDraw.Draw(im)
#         scr = scores[i]
#         lab = labels[i][scr > thrh]
#         box = boxes[i][scr > thrh]
#         scrs = scores[i][scr > thrh]
#         for j,b in enumerate(box):
#             draw.rectangle(list(b), outline='red',)
#             draw.text((b[0], b[1]), text=f"label: {lab[j].item()} {round(scrs[j].item(),2)}", font=ImageFont.load_default(), fill='blue')
#         if path == "":
#             im.save(f'results_{i}.jpg')
#         else:
#             im.save(path)
import random
from PIL import Image, ImageDraw, ImageFont

def draw(images, labels, boxes, scores, thrh=0.6, path=""):
    """
    在图像上绘制检测框和标签，每个框使用随机颜色。
    
    Args:
        images: PIL Image 对象列表
        labels: 标签列表 [batch_size, num_boxes]
        boxes: 边界框列表 [batch_size, num_boxes, 4] (x1, y1, x2, y2)
        scores: 置信度分数列表 [batch_size, num_boxes]
        thrh: 置信度阈值
        path: 保存路径。如果为空，则保存为 results_i.jpg
    """
    import random
    from PIL import ImageDraw, ImageFont
    import colorsys

    # 假设 images, scores, labels, boxes 和 thrh 已定义
    # 类别颜色映射表
    category_colors = {}

    for i, im in enumerate(images):
        # 创建可绘制对象
        
        draw = ImageDraw.Draw(im)
        
        # 获取当前图像的预测结果，并根据阈值过滤
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]
        
        # 为当前图像中的每个检测框绘制
        for j, b in enumerate(box):
            category = lab[j].item()
            
            if category not in category_colors:
                # 如果该类别还没有分配颜色，则生成新的颜色
                hue = random.random()  # 色调 0~1
                saturation = 0.8 + random.random() * 0.2  # 饱和度 0.8~1.0
                value = 0.8 + random.random() * 0.2       # 明度 0.8~1.0
                
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                color = tuple(int(c * 255) for c in rgb)
                
                # 将新颜色添加到映射表
                category_colors[category] = color
            else:
                # 使用已分配的颜色
                color = category_colors[category]
            
            hex_color = '#{:02x}{:02x}{:02x}'.format(*color)  # 转为十六进制，用于PIL
            
            # 绘制矩形框
            draw.rectangle(list(b), outline=hex_color, width=5)  # width 可选，让框更明显
            
            # 添加标签文本
            text = f"{category}/\\{round(scrs[j].item(), 2)}"
            if b[1] - 35 >= 35:
                text_y = b[1] - 35
            else:
                text_y = b[1] + 35
            draw.text((b[0], text_y), 
                      text=text, 
                      font=ImageFont.truetype("DejaVuSans.ttf", size=25), fill='white', stroke_fill=hex_color, 
                      stroke_width=5)        
        # 保存图像
        if path == "":
            im.save(f'results_{i}.jpg')
        else:
            # 如果 path 是单个文件名，建议改为 path_{i}.jpg 避免覆盖
            save_path = path if len(images) == 1 else f"{path}_{i}.jpg"
            im.save(save_path)

# --- 使用示例 ---
# 假设你有以下数据：
# images: [PIL.Image, PIL.Image, ...]
# labels: [torch.Tensor([1, 2, 1]), torch.Tensor([0, 2]), ...]
# boxes:  [torch.Tensor([[x1,y1,x2,y2], ...]), ...]
# scores: [torch.Tensor([0.9, 0.7, 0.5]), ...]

# draw(images, labels, boxes, scores, thrh=0.6, path="output")
            
def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')
    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
    
    model = Model().to(args.device)
    im_pil = Image.open(args.im_file).convert('RGB')
    new_size = (im_pil.width * 2, im_pil.height * 2)
    im_pil = im_pil.resize(new_size, Image.Resampling.LANCZOS)  # 使用高质量重采样
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)
    
    # # 将图像放大2倍
    # new_size = (im_org.width * 2, im_org.height * 2)
    # im_pil = im_org.resize(new_size, Image.Resampling.LANCZOS)  # 使用高质量重采样
    
    transforms = T.Compose([
        T.Resize((640, 640)),  
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None].to(args.device)
    if args.sliced:
        num_boxes = args.numberofboxes
        
        aspect_ratio = w / h
        num_cols = int(np.sqrt(num_boxes * aspect_ratio)) 
        num_rows = int(num_boxes / num_cols)
        slice_height = h // num_rows
        slice_width = w // num_cols
        overlap_ratio = 0.2
        slices, coordinates = slice_image(im_pil, slice_height, slice_width, overlap_ratio)
        predictions = []
        for i, slice_img in enumerate(slices):
            slice_tensor = transforms(slice_img)[None].to(args.device)
            with autocast():  # Use AMP for each slice
                output = model(slice_tensor, torch.tensor([[slice_img.size[0], slice_img.size[1]]]).to(args.device))
            torch.cuda.empty_cache() 
            labels, boxes, scores = output
            
            labels = labels.cpu().detach().numpy()
            boxes = boxes.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()
            predictions.append((labels, boxes, scores))
        
        merged_labels, merged_boxes, merged_scores = merge_predictions(predictions, coordinates, (h, w), slice_width, slice_height)
        labels, boxes, scores = postprocess(merged_labels, merged_boxes, merged_scores)
    else:
        output = model(im_data, orig_size)
        labels, boxes, scores = output
        
    draw([im_pil], labels, boxes, scores, 0.3)
  
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-f', '--im-file', type=str, )
    parser.add_argument('-s', '--sliced', type=bool, default=True)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-nc', '--numberofboxes', type=int, default=25)
    args = parser.parse_args()
    main(args)
