import torch
import cv2
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from imgaug import augmenters as iaa
from torchvision import transforms

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


class Music_Dataset(Dataset):

    def __init__(self, xml_list, len_data):
        self.xml_list = xml_list
        self.len_data = len_data
        self.to_tensor = transforms.ToTensor()
        self.resize = iaa.Resize({"shorter-side": 200, "longer-side": "keep-aspect-ratio"})
        self.flip = iaa.Fliplr(0.5)

    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):

        bbox_dir = r'bbox'
        image_dir = r'images'
        label_dir = r'label'

        label_list = os.listdir(label_dir)
        bbox_list = os.listdir(bbox_dir)
        image_list = os.listdir(image_dir)

        image = Image.open(image_dir + '/' + image_list[idx])
        image = np.array(image)

        bbox = np.load(bbox_dir + '/' + bbox_list[idx])
        bbox = bbox.reshape(1, bbox.shape[0], bbox.shape[1])
        image, bbox = self.resize(image=image, bounding_boxes=bbox)

        '''
        for _bbox in bbox[0]:
            cv2.rectangle(image, (_bbox[0], _bbox[1]), (_bbox[2], _bbox[3]),
                          color=(0, 255, 0), thickness=1)
        cv2.imshow('test', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        '''

        bbox = bbox.squeeze(0).tolist()
        image = self.to_tensor(image)
        labels = np.load(label_dir + '/' + label_list[idx])
        labels -= 1
        targets = []

        d = {'boxes': torch.tensor(bbox, device=device),
             'labels': torch.tensor(labels, dtype=torch.int64, device=device)}

        targets.append(d)
        return image, targets


