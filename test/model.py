# https://smp.readthedocs.io/en/latest/index.html
# https://smp.readthedocs.io/en/latest/encoders_timm.html
# timm encoder 쓸땐 encoder name에 tu- 붙이기

# code based on https://smp.readthedocs.io/en/latest/insights.html#

import torch
import segmentation_models_pytorch as smp

def getModel():

	model = smp.PAN(
			encoder_name="tu-xception41",
			encoder_weights="imagenet",
			in_channels=3,
			classes=15
		)
	

	# model = smp.Unet(
  #        encoder_name="resnet18",
  #        encoder_weights="imagenet",
  #        in_channels=3,
  #        classes=15
  #     )

	return model