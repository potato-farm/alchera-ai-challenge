import torch
import numpy as np
from tqdm import tqdm
import albumentations as A

def test(model, data_loader, device, size):
	model.to(device)
	transform = A.Compose([A.Resize(size, size)])
	print('Start prediction.')
	
	model.eval()
	
	file_name_list = []
	preds_array = np.empty((0, size*size), dtype=np.compat.long)
	
	with torch.no_grad():
		for step, (imgs, image_infos) in enumerate(tqdm(data_loader)):
			
			# inference (512 x 512)
			outs = model(torch.stack(imgs).to(device))
			oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
			preds_array = np.array(oms)
			
			file_name_list.append([i['file_name'] for i in image_infos])
	print("End prediction.")
	file_names = [y for x in file_name_list for y in x]
	
	return file_names, preds_array


import pandas as pd
import os
import cv2
def saveSubmission(file_names, preds_array, output_path, custom_name):
	
	for file_name, mask in zip(file_names, preds_array):
		save_dir = os.path.join(output_path, file_name) #file name: without extension
		save_dir += ".png"
		cv2.imwrite(save_dir, mask)