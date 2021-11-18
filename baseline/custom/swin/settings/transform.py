import albumentations as A
from albumentations.pytorch import ToTensorV2
import ttach as tta
# https://albumentations-demo.herokuapp.com/

def getTransform():

  train_transform = A.Compose([
                              A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=0.5),
                              A.GaussNoise(p=0.3),
                              A.OneOf([
                                  A.Blur(p=1.0),
                                  A.GaussianBlur(p=1.0),
                                  A.MedianBlur(blur_limit=5, p=1.0),
                                  A.MotionBlur(p=1.0),
                              ], p=0.1),
                              A.Resize(512, 512, p=1.0),
                              A.Normalize(), 
                              ToTensorV2()
                              ])

  val_transform = A.Compose([
                            A.Resize(512, 512, p=1.0),
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2(),
                            ])

  return train_transform, val_transform


def getInferenceTransform():

  test_transform = A.Compose([
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2(),
                            ])
  tta_transform = tta.Compose(
    [
        tta.Scale(scales=[1, 2, 4]),
        tta.Multiply(factors=[0.9, 1, 1.1]),        
    ]
)

                        
  return test_transform, tta_transform        