'''
Code Reference : https://github.com/HyungjoByun/Projects/blob/main/Faster%20RCNN/FasterRCNN_Train.ipynb
'''
import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

backbone = torchvision.models.vgg16(pretrained=True).features[:-1]
backbone_out = 512
backbone.out_channels = backbone_out

anchor_generator = AnchorGenerator(sizes=((32, 64, 128),), aspect_ratios=((0.25, 0.5, 1.0, 2.0, 4.0),))

resolution = 7
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=resolution, sampling_ratio=2)

box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(in_channels=backbone_out * (resolution ** 2),
                                                               representation_size=512)
box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(512, 37)  # 37개 class

model = torchvision.models.detection.FasterRCNN(backbone, num_classes=None,
                                                min_size=100, max_size=1000,
                                                rpn_anchor_generator=anchor_generator,
                                                rpn_pre_nms_top_n_train=3000, rpn_pre_nms_top_n_test=1000,
                                                rpn_post_nms_top_n_train=1000, rpn_post_nms_top_n_test=150,
                                                rpn_nms_thresh=0.7, rpn_fg_iou_thresh=0.5, rpn_bg_iou_thresh=0.5,
                                                rpn_batch_size_per_image=128, rpn_positive_fraction=0.5,
                                                box_roi_pool=roi_pooler, box_head=box_head, box_predictor=box_predictor,
                                                box_score_thresh=0.7, box_nms_thresh=0.7, box_detections_per_img=30,
                                                box_fg_iou_thresh=0.7, box_bg_iou_thresh=0.7,
                                                box_batch_size_per_image=128, box_positive_fraction=0.25
                                                )
# roi head 있으면 num_class = None으로 함

for param in model.rpn.parameters():
    torch.nn.init.normal_(param, mean=0.0, std=0.01)

for name, param in model.roi_heads.named_parameters():
    if "bbox_pred" in name:
        torch.nn.init.normal_(param, mean=0.0, std=0.001)
    elif "weight" in name:
        torch.nn.init.normal_(param, mean=0.0, std=0.01)
    if "bias" in name:
        torch.nn.init.zeros_(param)
