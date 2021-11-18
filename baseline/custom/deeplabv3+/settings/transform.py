import albumentations as A
from albumentations.pytorch import ToTensorV2
import ttach as tta
# https://albumentations-demo.herokuapp.com/

def getTransform():

  train_transform = A.Compose([
                              A.Resize(512, 512, p=1.0),
                              A.GaussNoise(p=0.3),
                              A.OneOf([
                                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=1.0),
                                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, p=1.0)
                              ], p=0.5),
                              A.OneOf([
                                  A.Blur(p=1.0),
                                  A.GaussianBlur(p=1.0),
                                  A.MedianBlur(blur_limit=5, p=1.0),
                                  A.MotionBlur(p=1.0),
                              ], p=0.1),
                              A.Rotate(limit=(-20,20), p=0.3),
                              A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        tta.Scale(scales=[0.5, 0.75, 1, 1.25, 1.5]),
        tta.Multiply(factors=[0.9, 1, 1.1]),        
    ]
)

                        
  return test_transform, tta_transform        