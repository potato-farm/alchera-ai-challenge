import torch
import numpy as np
from tqdm import tqdm
import albumentations as A
import pandas as pd
import os
import cv2

def test_and_save(model, data_loader, device, output_path, custom_name):
	model.to(device)
	print('Start prediction.')
	
	model.eval()
	
	file_name_list = []
	#preds_array = np.empty((0, size*size), dtype=np.compat.long)
	
	with torch.no_grad():
		for step, (imgs, image_names) in enumerate(tqdm(data_loader)):
			h, w = imgs[0].shape[1], imgs[0].shape[2]			
			resizedImg = torch.nn.functional.interpolate(imgs[0].unsqueeze(dim=0), size=(1280, 960), mode='area')
			outs = model(resizedImg.to(device))
			mask = torch.argmax(outs.squeeze(), dim=0).detach().cpu().numpy()
			mask = cv2.resize(mask.astype('float32'), dsize=(w, h), interpolation=cv2.INTER_AREA) 
			save_dir = f"{os.path.join(output_path, image_names[0])}.png"
			cv2.imwrite(save_dir, mask)
			# # for making colored mask image
			# mask2color = cv2.applyColorMap(cv2.cvtColor(mask.astype('uint8'), cv2.COLOR_GRAY2RGB) * 17, cv2.COLORMAP_SUMMER)
			# save_color_dir = f"{os.path.join(output_path, image_names[0])}_color.png"
			# cv2.imwrite(save_color_dir, mask2color)
	print("End prediction.")