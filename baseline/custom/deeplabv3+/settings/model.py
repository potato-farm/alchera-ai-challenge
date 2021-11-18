import segmentation_models_pytorch as smp
import torch

# https://smp.readthedocs.io/en/latest/index.html
# https://smp.readthedocs.io/en/latest/encoders_timm.html
# timm encoder 쓸땐 encoder name에 tu- 붙이기

def getModel():
	
	model = smp.DeepLabV3Plus(
			encoder_name="resnet34",
			encoder_weights="imagenet",
			in_channels=3,
			classes=15
		)
	model_path = "/opt/ml/input/data/alchera-ai-challenge/output/deeplabv3_2/models/epoch29.pth"
	checkpoint = torch.load(model_path)
	model.load_state_dict(checkpoint['model'])
	
	return model