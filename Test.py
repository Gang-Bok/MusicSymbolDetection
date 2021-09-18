import torch
import os
import pickle
import cv2
import numpy as np
from FasterRCNN import model
from Muscima_Dataset import Music_Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

xml_path = r'muscima-pp/v2.1/data/annotations'
xml_list = os.listdir(xml_path)
model_path = r'Faster_RCNN/Check_point.pth'
len_data = 805
start_idx = 0

model.load_state_dict(torch.load(model_path))
model.to(device)

'''
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
'''

dataset = Music_Dataset(xml_list[:len_data], len_data - start_idx)
dataloader = DataLoader(dataset, batch_size=1)

with open('label_dic.pickle', 'rb') as fw:
    label_dic = pickle.load(fw)

new_label_dic = {}
for idx, _key in enumerate(label_dic.keys()):
    new_label_dic[idx] = _key
print(label_dic)
print(new_label_dic)

for i, (image, targets) in enumerate(dataloader, start_idx):
    model.eval()
    targets[0]['boxes'].squeeze_(0)
    targets[0]['labels'].squeeze_(0)
    # with profiler.profile(with_stack=True, use_cuda=True, profile_memory=True) as prof:
    image_copy = image.cpu().numpy()
    outputs = model(image.to(device))
    bbox = outputs[0]['boxes']
    labels = outputs[0]['labels']
    new_img = np.moveaxis(image_copy[0]*255, 0, -1)
    new_img = new_img.astype(np.uint8)
    new_img = cv2.UMat(new_img)
    for _labels in labels:
        print(new_label_dic[int(_labels.cpu())])
    for _bbox in bbox:
        _bbox = _bbox.cpu().detach().numpy().astype(np.uint32)
        print(_bbox)
        cv2.rectangle(new_img, (_bbox[0], _bbox[1]), (_bbox[2], _bbox[3]), (0, 255, 0), 1)
    cv2.imshow('1', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

