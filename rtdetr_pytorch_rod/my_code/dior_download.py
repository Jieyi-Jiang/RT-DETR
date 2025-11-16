import openxlab

# 进行登录，输入对应的AK/SK
openxlab.login( 
    ak="zdmbxgbld3xnnoo9zao3", 
    sk="qagw45vxej1aepk3qolymbbbe8z0lbdborowmjyr") 

from openxlab.dataset import info
info(dataset_repo='OpenDataLab/DIOR') #数据集信息及文件列表查看

from openxlab.dataset import get
get(dataset_repo='OpenDataLab/DIOR', target_path='/root/autodl-tmp/dior')  # 数据集下载

from openxlab.dataset import download
download(dataset_repo='OpenDataLab/DIOR',source_path='/README.md', target_path='/root/autodl-tmp/dior') #数据集文件下载

