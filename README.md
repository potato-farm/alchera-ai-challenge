# alchera-ai-challenge

</br>

## ๐จโ๐พ Team

- ๋ฌด๋ญ๋ฌด๋ญ ๊ฐ์๋ฐญ
- ํ ๊ตฌ์ฑ์ : ๊น์ธ์, ๋ฐ์ฑ์ง, ์ ์นํ, ์ด์์, ์ด์ฑ์ค, ์กฐ์ฑ์ฑ

</br>

## ๐ LB Score

- LB : 0.6732 mIoU 4th

</br>

## ๐ Project Summary

- ์ฌ๋์ ์ฌ์ง์ ์๋ ฅ๊ฐ์ผ๋ก ๋ฐ์ ๊ฐ ์ ์ฒด๋ถ์ ๋ณ๋ก Semantic Segmentation
- Train : Pytorch๊ธฐ๋ฐ์ Baseline / [Segmentation Models Pytorch](https://github.com/qubvel/segmentation_models.pytorch)๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ฅผ ์ด์ฉ
- Test : ๋ณ๋์ ํ์ผ๋ก ๋ถ๋ฆฌ (์ ์ถ ์ฉ๋ ์ ํ : 100mb)
- EDA: Train ๋ฐ์ดํฐ์ ์ค์  Test ๋ฐ์ดํฐ์ ํน์ง ํ์
- Data Augmentation : Albumentation ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ฅผ ์ด์ฉ
    - Rotate(20)
    - Brightness/Contrast, HueSaturation
    - Blur (Gaussian, Median, Motion)
    - GaussNoise
- Ensemble : Soft voting ๋ฐฉ์ ํ์ฉ
    
</br>

## ๐บ Project Results

![project-result](https://user-images.githubusercontent.com/41667491/144161189-9fa5287e-f617-46cd-a497-3e9aa64f81c2.gif)


</br>

## โ Development Environment
- GPU : Nvidia Tesla V100 
- OS : Linux Ubuntu 18.04
- Runtime : Python 3.8.5
- Main Dependency : segmentation_models.pytorch, Pytorch 1.7.1, OpenCV 4.5.1
