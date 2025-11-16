import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 加载评估结果
coco_eval_result = torch.load('../output/rtdetr_r50vd_6x_coco/eval.pth')

print(type(coco_eval_result))

# 查看具体内容
print(coco_eval_result["date"])

# 如果是字典，查看键
if isinstance(coco_eval_result, dict):
    print(coco_eval_result.keys())


# 如果需要重新格式化结果
# 这通常是COCOeval.evaluate()和COCOeval.accumulate()之后的结果

# 打印详细评估报告
# 注意：这需要原始的COCOeval对象
# cocoEval = COCOeval(...)
# cocoEval.eval = coco_eval_result
# cocoEval.params.imgIds = imgIds
# cocoEval.accumulate()
# cocoEval.summarize()