# alchera-ai-challenge

</br>

## 👨‍🌾 Team

- 무럭무럭 감자밭
- 팀 구성원 : 김세영, 박성진, 신승혁, 이상원, 이채윤, 조성욱

</br>

## 🏆 LB Score

- LB : 0.6732 mIoU 4th

</br>

## 🔑 Project Summary

- 사람의 사진을 입력값으로 받아 각 신체부의 별로 Semantic Segmentation
- Train : Pytorch기반의 Baseline / [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch)라이브러리를 이용
- Test : 별도의 파일로 분리 (제출 용량 제한 : 100mb)
- EDA: Train 데이터와 실제 Test 데이터의 특징 파악
- Data Augmentation : Albumentation 라이브러리를 이용
    - Rotate(20)
    - Brightness/Contrast, HueSaturation
    - Blur (Gaussian, Median, Motion)
    - GaussNoise
- Ensemble : Soft voting 방식 활용
    
</br>

## 🕺 Project Results

![project-result](https://user-images.githubusercontent.com/41667491/144161189-9fa5287e-f617-46cd-a497-3e9aa64f81c2.gif)


</br>

## ⚙ Development Environment
- GPU : Nvidia Tesla V100 
- OS : Linux Ubuntu 18.04
- Runtime : Python 3.8.5
- Main Dependency : segmentation_models.pytorch, Pytorch 1.7.1, OpenCV 4.5.1
