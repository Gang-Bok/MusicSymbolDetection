# MusicSymbolDetection

음악 기호 데이터를 이용하여 Object Detection 방식으로 사람이 손 글씨로 악보를 그렸을 때  어느 위치에 어떤 음악 기호를 그렸는지 알려주는 모델을 구축하였다.

Using in this Application : https://github.com/Senitf/MusicNote_SwiftUI_YOLOv5

## HOW TO USE
1. Dataset.zip 파일과 yolov5폴더에 있는 파일들을 clone을 통해 받는다.
2. https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data 사이트를 통해 Custom Data를 train 시키고 모델을 사용한다.

## Issue

|날짜|내용
|---|---|
|2021.09.18|Faster-RCNN을 이용하여 Muscima_pp data를 Train 하였음
|2021.10.17|Faster-RCNN을 torch.jit.trace를 이용해 Swift에서 사용할 수 있는 형태로 만들려고 했으나 지원을 하지 않는 것 같아 YOLOv5모델을 사용하는 것으로 변경함 

## References
Dataset : https://github.com/OMR-Research/muscima-pp

Faster-RCNN :  https://hyungjobyun.github.io/machinelearning/FasterRCNN2/

Yolo: https://github.com/ultralytics/yolov5


