import torch
import argparse
import os
import shutil
from importlib import import_module

from dataset.base_dataset import CustomDataset
from utils.train_method import train
from utils.set_seed import setSeed

def getArgument():
	# Custom 폴더 내 훈련 설정 목록을 선택
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',type=str ,required=True)
	return parser.parse_known_args()[0].dir

def main(custom_dir):

	arg = getattr(import_module(f"{custom_dir}.settings.arg"), "getArg")()

	device = "cuda" if torch.cuda.is_available() else "cpu"
	setSeed(arg.seed)

	train_transform, val_transform = getattr(import_module(f"{custom_dir}.settings.transform"), "getTransform")()

	train_dataset = CustomDataset(image_root=arg.train_image_root, mask_root=arg.train_mask_root, mode='train', transform=train_transform)
	val_dataset = CustomDataset(image_root=arg.val_image_root, mask_root=arg.val_mask_root, mode='val', transform=val_transform)

	trainLoader, valLoader = getattr(import_module(f"{custom_dir}.settings.dataloader"), "getDataloader")(
		train_dataset, val_dataset, arg.batch, arg.train_worker, arg.valid_worker)

	model = getattr(import_module(f"{custom_dir}.settings.model"), "getModel")()
	criterion = getattr(import_module(f"{custom_dir}.settings.loss"), "getLoss")()
	optimizer, scheduler = getattr(import_module(f"{custom_dir}.settings.opt_scheduler"), "getOptAndScheduler")(model, arg.lr)

	outputPath = os.path.join(arg.output_path, arg.custom_name)

	# output Path 내 설정 저장
	shutil.copytree(f"{custom_dir}",outputPath)
	os.makedirs(outputPath+"/models")
	
	# wandb
	if arg.wandb:
		from utils.wandb_method import WandBMethod
		WandBMethod.login(arg, model, criterion)

	train(arg.epoch, model, trainLoader, valLoader, criterion, optimizer,scheduler, outputPath, arg.save_capacity, device, arg.wandb)

if __name__=="__main__":
	custom_dir = getArgument()
	main(custom_dir)