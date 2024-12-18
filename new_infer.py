import os
import time
import datetime
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from numpy import mean
import torch.nn.functional as F
import torchvision.utils as utils
from config import *
from misc import *
import sys
import cv2
torch.manual_seed(2021)
sys.path.append('../')
from metric_caller import CalTotalMetric
from excel_recorder import MetricExcelRecorder
#ACC
def main(exp_name,net,scale,results_path,pth_path):
    check_mkdir(results_path)
    to_test = OrderedDict([
        ('CAMO', camo_path),
        ('CHAMELEON', chameleon_path),
        ('COD10K', cod10k_path),
        ('NC4K', nc4k_path)
    ])
    results = OrderedDict()
    img_transform = transforms.Compose([
        transforms.Resize((scale,scale)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    to_pil = transforms.ToPILImage()
    net.load_state_dict(torch.load(pth_path))
    print('Load {} succeed!'.format(exp_name+'.pth'))
    net.eval()
    path_excel = './results_excel1.xlsx'
    with torch.no_grad():
        excel_logger = MetricExcelRecorder(xlsx_path=path_excel, dataset_names=[name for name, root in to_test.items()])
        start = time.time()
        for name, root in to_test.items():
            cal_total_seg_metrics = CalTotalMetric()
            time_list = []
            image_path = os.path.join(root, 'Imgs')
            mask_path = os.path.join(root, 'GT')
            check_mkdir(os.path.join(results_path, exp_name, name))
            img_suffix = 'jpg'
            img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith(img_suffix)]
            for idx, img_name in enumerate(img_list):
                img = Image.open(os.path.join(image_path, img_name + '.' + img_suffix)).convert('RGB')

                mask = np.array(Image.open(os.path.join(mask_path, img_name + '.png')).convert('L'))

                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda()

                start_each = time.time()

                predictions = net(img_var)
                prediction = predictions[-1]
                prediction = torch.sigmoid(prediction)
                time_each = time.time() - start_each
                time_list.append(time_each)

                prediction = np.array(transforms.Resize((h, w))(to_pil(prediction.data.squeeze(0).cpu())))
                Image.fromarray(prediction).convert('L').save(
                    os.path.join(results_path, exp_name, name, img_name + '.png'))

                cal_total_seg_metrics.step(prediction, mask, mask_path)
            print(('{}'.format(exp_name)))
            print("{}'s average Time Is : {:.3f} s".format(name, mean(time_list)))
            print("{}'s average Time Is : {:.1f} fps".format(name, 1 / mean(time_list)))
            results = cal_total_seg_metrics.get_results()
            excel_logger(row_data=results, dataset_name=name, method_name=exp_name)
            print(results)
    end = time.time()
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))


def evluation_with_resultspath(results_path,path_excel):
    print(results_path)
    _, exp_name = os.path.split(results_path)
    to_test = OrderedDict([
        ('CAMO', camo_path),
        ('CHAMELEON', chameleon_path),
        ('COD10K', cod10k_path),
        ('NC4K', nc4k_path)
    ])
    excel_logger = MetricExcelRecorder(xlsx_path=path_excel, dataset_names=[name for name, root in to_test.items()])

    for name, root in to_test.items():
        print(os.path.join(results_path,name))
        if not os.path.exists(os.path.join(results_path,name)):
            continue
        print(name)
        cal_total_seg_metrics = CalTotalMetric()
        image_path = os.path.join(root, 'Imgs')
        mask_path = os.path.join(root, 'GT')
        img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('jpg')]
        for idx, img_name in enumerate(img_list):
            result_img_path = os.path.join(results_path, name, img_name + '.png')
            # if not os.path.exists(result_img_path):
            #     os.rename(os.path.join(results_path, name, img_name + '.tif'), result_img_path)
            prediction = Image.open(result_img_path).convert('L')

            mask = Image.open(os.path.join(mask_path, img_name + '.png')).convert('L')
            if not prediction.size == mask.size:
                mask = mask.resize(prediction.size)
            cal_total_seg_metrics.step(np.array(prediction), np.array(mask), mask_path)

        results = cal_total_seg_metrics.get_results()
        print(results)
        excel_logger(row_data=results, dataset_name=name, method_name=exp_name)

def evaluation_COD(exp_name,net,scale,results_path,pth_path):
    main(exp_name,net,scale,results_path,pth_path)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    exp_name = 'PAFNet'
    from SARNet import SARNet

    net = SARNet('pvt_v2_b3').cuda()
    net = torch.nn.DataParallel(net)
    net = net.cuda()
    net.eval()

    pth_path = os.path.join(root,'ckpt/PAFNet/150.pth')
    results_path = os.path.join(root,'PAFNet/results')
    #evluation_with_resultspath(results_path,'./results_1.xlsx')
    main(exp_name, net, 384, results_path, pth_path)
