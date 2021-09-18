'''
Code Reference : https://github.com/HyungjoByun/Projects/blob/main/Faster%20RCNN/FasterRCNN_Train.ipynb
'''
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import torch.autograd.profiler as profiler
from FasterRCNN import model
from Muscima_Dataset import Music_Dataset
import numpy as np
import cv2

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')


def Total_Loss(loss):
    loss_objectness = loss['loss_objectness']
    loss_rpn_box_reg = loss['loss_rpn_box_reg']
    loss_classifier = loss['loss_classifier']
    loss_box_reg = loss['loss_box_reg']

    rpn_total = loss_objectness + 10 * loss_rpn_box_reg
    fast_rcnn_total = loss_classifier + 1 * loss_box_reg

    total_loss = rpn_total + fast_rcnn_total

    return total_loss


writer = SummaryWriter(r"Faster_RCNN")
xml_path = r'muscima-pp/v2.1/data/annotations'
xml_list = os.listdir(xml_path)
total_epoch = 10

len_data = 805
term = 100

loss_sum = 0

model.to(device)

optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch, eta_min=0.00001)

try:
    check_point = torch.load(r"Faster_RCNN/Check_point.pth")
    start_epoch = check_point['epoch']
    start_idx = check_point['iter']
    model.load_state_dict(check_point['state_dict'])
    optimizer.load_state_dict(check_point['optimizer'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch, eta_min=0.00001,
                                                           last_epoch=start_epoch)
    scheduler.load_state_dict(check_point['scheduler'])

    if start_idx == len_data:
        start_idx = 0
        start_epoch = start_epoch + 1

except:
    print("check point load error!")
    start_epoch = 0
    start_idx = 0

print("start_epoch = {} , start_idx = {}".format(start_epoch, start_idx))

print("Training Start")
model.train()
start = time.time()

for epoch in range(start_epoch, total_epoch):

    writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

    dataset = Music_Dataset(xml_list[:len_data], len_data - start_idx)
    dataloader = DataLoader(dataset, batch_size=1)

    for i, (image, targets) in enumerate(dataloader, start_idx):

        optimizer.zero_grad()

        targets[0]['boxes'].squeeze_(0)
        targets[0]['labels'].squeeze_(0)

        '''
        image_copy = image.cpu().numpy()
        new_img = np.moveaxis(image_copy[0] * 255, 0, -1)
        new_img = new_img.astype(np.uint8)
        new_img = cv2.UMat(new_img)
        cnt = 1
        for _bbox in targets[0]['boxes']:
            print(_bbox)
            _bbox = _bbox.cpu().detach().numpy().astype(np.uint32)
            print(_bbox)
            cnt += 1
            cv2.rectangle(new_img, (_bbox[0], _bbox[1]), (_bbox[2], _bbox[3]), (0, 255, 0), 1)
        cv2.imshow('1', new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''

        loss = model(image.to(device), targets)
        total_loss = Total_Loss(loss)
        loss_sum += total_loss

        if (i + 1) % term == 0:
            end = time.time()
            print("Epoch {} | Iter {} | Loss: {} | Duration: {} min".format(epoch, (i + 1), (loss_sum / term).item(),
                                                                            int((end - start) / 60)))
            writer.add_scalar('Training Loss', loss_sum / term, epoch * len_data + i)

            state = {
                'epoch': epoch,
                'iter': i + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(state, r'Faster_RCNN\Check_point.pth')

            loss_sum = 0
            start = time.time()

        total_loss.backward()
        optimizer.step()

    start_idx = 0
    scheduler.step()

    state = {
        'epoch': epoch,
        'iter': i + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, r'Faster_RCNN/Check_point.pth')

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), r'Faster_RCNN/Check_point.pth'.format(epoch))




