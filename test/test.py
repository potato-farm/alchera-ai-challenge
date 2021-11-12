import torch
from torch.utils.data.dataloader import DataLoader
import os

MODELPATH = "./best.pth" #모델 이름
WORKER = 4 
TTA = False
SIZE = (1024,1024) #resize해서 확인할건데 모델 훈련한 크기랑 같은 크기 맞추는거 추천

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def getArgument():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type=str, required=True)
	parser.add_argument('--save_path', type=str, required=True)

	args = parser.parse_known_args()[0]

	return args.data_path, args.save_path

def loadDataset(dataPath):
	from base_dataset import CustomDataset
	from albumentations.pytorch import ToTensorV2
	import albumentations as A

	dataset = CustomDataset(image_root=dataPath, mode="test",
		transform=A.Compose([
			A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			ToTensorV2(),
  	]))
	
	return dataset

def loadModelWeight():
	from model import getModel
	model = getModel()
	model.load_state_dict(torch.load(MODELPATH,map_location=DEVICE)['model'])
	return model

def wrapTTAModel(model):
	import ttach as tta
	tta_transform = tta.Compose(
    [
        # tta.HorizontalFlip(), #개구림
        # tta.Rotate90(angles=[0, 180]), #구릴듯
        tta.Multiply(factors=[0.9, 1, 1.1]),    #밝기    
    ])
	
	return tta.SegmentationTTAWrapper(model, tta_transform)

def inference(model,dataloader,savePath):
	from tqdm import tqdm
	import cv2
	model.to(DEVICE)
	model.eval()

	with torch.no_grad():
		for image, imageName in tqdm(dataloader):
			h,w = image.shape[2:] # b c h w
			image = torch.nn.functional.interpolate(image,size=SIZE,mode='bilinear',align_corners=True)
			out = model(image.to(DEVICE))
			out = torch.nn.Upsample(size=(h,w),mode='bilinear',align_corners=True)(out)
			mask = torch.argmax(out.squeeze(),dim=0).detach().cpu().numpy()
			
			for i in range(14,0,-1): # TODO 이거 시각화용임 제출할땐 뺴야댐
				mask[mask==i] = i*14

			cv2.imwrite(os.path.join(savePath,imageName[0]+".png"),mask)
			
if __name__=="__main__":
	dataPath, savePath = getArgument()
	dataset = loadDataset(dataPath)
	dataloader = DataLoader(dataset, batch_size=1, num_workers=WORKER, shuffle= False)

	model = loadModelWeight()
	if TTA:
		model = wrapTTAModel(model)
	
	os.makedirs(savePath, exist_ok=True)
	
	inference(model, dataloader, savePath)

 