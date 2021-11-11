from math import sqrt, floor
import wandb
import random
import cv2
import numpy as np

class WandBMethod:
	'''
	WandB에 관련된 method들이 모여있는 helper 클래스 입니다.
	'''

	@staticmethod
	def login(arg, model, criterion, log="gradients", log_freq=10):
		'''
		초기화 함수 모음
		'''
		wandb.login()
		wandb.init(project=arg.wandb_project, entity=arg.wandb_entity, name=arg.custom_name, config=arg)
		wandb.watch(model,criterion, log=log,log_freq=log_freq)
	
	@staticmethod
	def trainLog(loss, acc, lr):
		'''
		train에서 각 batch마다 보내는 정보로, mmsegmentation format에 최대한 맞춤
		'''
		wandb.log({"train/loss":loss.item(),"train/decode.loss_ce":loss.item(),"train/decode.acc_seg":acc.item(), "learning_rate_torch":lr})
	
	@classmethod
	def validLog(cls, clsIoU, clsAcc,clsMeanAcc, mAcc, mIoU, images, outputs, masks):
		'''
		각 epoch당 valid가 끝난 뒤 한번 보내지는 정보들로, mmsegmentation format에 최대한 맞춰서 정보가 좀 직관적이지 않고 많음
		'''
		categoryDict = {i:category for i, category in enumerate([
			'Background', 'Body', 'RightHand', 'LeftHand', 'LeftFeet', 'RightFeet', 
			'RightThigh', 'LeftThigh', 'RightCalf', 'LeftCalf', 'LeftArm', 'RightArm', 
			'LeftForeArm', 'RightForeArm','Head'])}
		
		image = cls.concatImages(images)
		output = cls.concatImages(outputs)
		mask = cls.concatImages(masks)
		
		wandb.log({
			"val/IoU.Background":clsIoU[0],
			"val/IoU.Body":clsIoU[1],
			"val/IoU.RightHand":clsIoU[2],
			"val/IoU.LeftHand":clsIoU[3],
			"val/IoU.RightFeet":clsIoU[5],
			"val/IoU.LeftFeet":clsIoU[4],
			"val/IoU.RightThigh":clsIoU[6],
			"val/IoU.LeftThigh":clsIoU[7],
			"val/IoU.RightCalf":clsIoU[8],
			"val/IoU.LeftCalf":clsIoU[9],
			"val/IoU.RightArm":clsIoU[11],
			"val/IoU.LeftArm":clsIoU[10],
			"val/IoU.RightForeArm":clsIoU[13],
			"val/IoU.LeftForeArm":clsIoU[12],
			"val/IoU.Head":clsIoU[14],
	
			"val/Acc.Background":clsAcc[0],
			"val/Acc.Body":clsAcc[1],
			"val/Acc.RightHand":clsAcc[2],
			"val/Acc.LeftHand":clsAcc[3],
			"val/Acc.RightFeet":clsAcc[5],
			"val/Acc.LeftFeet":clsAcc[4],
			"val/Acc.RightThigh":clsAcc[6],
			"val/Acc.LeftThigh":clsAcc[7],
			"val/Acc.RightCalf":clsAcc[8],
			"val/Acc.LeftCalf":clsAcc[9],
			"val/Acc.RightArm":clsAcc[11],
			"val/Acc.LeftArm":clsAcc[10],
			"val/Acc.RightForeArm":clsAcc[13],
			"val/Acc.LeftForeArm":clsAcc[12],
			"val/Acc.Head":clsAcc[14],

			"val/aAcc":mAcc.item(),
			"val/mAcc":clsMeanAcc.item(), #wandb 맞추는중
			"val/mIoU":mIoU.item(),
			"image" : wandb.Image(image, masks={
					"predictions" : {
							"mask_data" : output,
							"class_labels":categoryDict
					},
					"ground_truth" : {
							"mask_data" : mask,
							"class_labels":categoryDict
					}}),
		})

	@staticmethod
	def pickImageStep(length):
		'''
		valid할 때 여러 batch 중 하나를 선택
		'''
		return random.randint(0,length-1)	

	@staticmethod
	def concatImages(images):
		'''
		여러개의 이미지를 하나의 이미지로 합쳐주는 과정입니다.
		batch size에서 가장 가까운 정사각형으로 설정되게 해놨습니다

		ex) 
		batch 8 -> 2x2 
		batch 16 -> 4x4
		batch 32 -> 5x5
		'''

		length = len(images)
		squareSide = floor(sqrt(length))

		hConcatImgs = []
		for i in range(0,squareSide*squareSide,squareSide):
			imgList = []
			for j in range(i,i+squareSide):
				if images[j].shape == (512,512): # mask의 경우 채널이 1개이기 때문에 따로 정제 필요
					nowImage = np.expand_dims(images[j],axis=2) 
				else:
					nowImage = np.transpose(images[j],(1,2,0)) # tensor -> cv2 포맷으로 변경
				imgList.append(nowImage)

			hConcatImgs.append(cv2.hconcat(imgList))

		fullImg = cv2.vconcat(hConcatImgs)
		resizedImg = cv2.resize(fullImg.astype('float32'), dsize=(512,512), interpolation=cv2.INTER_AREA) # 최종 출력을 512x512로 축소
		return resizedImg
