import segmentation_models_pytorch as smp
import torch

# https://smp.readthedocs.io/en/latest/index.html
# https://smp.readthedocs.io/en/latest/encoders_timm.html
# timm encoder 쓸땐 encoder name에 tu- 붙이기

def getModel():
	
	model = smp.DeepLabV3Plus(
			encoder_name="efficientnet-b4",
			encoder_weights="imagenet",
			in_channels=3,
			classes=15
		)

	model_path = "/opt/ml/input/data/alchera-ai-challenge/baseline/custom/deeplabv3+_eff/saved/.pth"
	checkpoint = torch.load(model_path)
	model.load_state_dict(checkpoint['model'])	
	return model