import torch
import argparse
import os
import shutil
from importlib import import_module
import ttach as tta
from dataset.base_dataset import CustomDataset
from utils.inference_method import test_and_save

def getArgument():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',type=str ,required=True)
	parser.add_argument('--model', type=str, required=True)
	parser.add_argument('--data_path', type=str, required=True)
	parser.add_argument('--save_path', type=str, required=True)

	args = parser.parse_known_args()[0]

	return args.dir, args.model, args.data_path, args.save_path


def main(custom_dir, model_name, image_dir, save_dir):
	
	device = "cuda" if torch.cuda.is_available() else "cpu"

	if os.path.isdir("./tempSettings"):
		shutil.rmtree("./tempSettings")
	shutil.copytree(f"../output/{custom_dir}/settings/", "./tempSettings/")

	arg = getattr(import_module(f"tempSettings.arg"), "getArg")()

	transfroms, ttaTransfrom = getattr(import_module(f"tempSettings.transform"), "getInferenceTransform")()
	dataset = CustomDataset(image_root=image_dir, mode="test", transform=transfroms)
	model = getattr(import_module(f"tempSettings.model"), "getModel")()
	dataloader = getattr(import_module(f"tempSettings.dataloader"), "getInferenceDataloader")(dataset, 1, arg.test_worker)

	stateDict = torch.load(f"../output/{custom_dir}/models/{model_name}.pth", map_location=device)
	model.load_state_dict(stateDict['model'])

	if arg.TTA:
		# tta 유무에 따라 모델 포장
	 	model = tta.SegmentationTTAWrapper(model, ttaTransfrom)

	os.makedirs(save_dir, exist_ok=True)
	test_and_save(model, dataloader, device, save_dir, arg.custom_name)

	shutil.rmtree("./tempSettings")

if __name__=="__main__":
	custom_dir, model_name, image_dir, save_dir = getArgument()
	main(custom_dir, model_name, image_dir, save_dir)