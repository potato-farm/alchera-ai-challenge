from easydict import EasyDict as eDict

def getArg():
	arg = eDict()

	arg.batch = 16
	arg.epoch = 75
	arg.lr = 5e-5
	arg.seed = 21
	arg.save_capacity = 5
	
	arg.train_image_root = "../input/train2014"
	arg.train_mask_root = "../input/train_mask"
	arg.val_image_root = "../input/val2014"
	arg.val_mask_root = "../input/val_mask"
	arg.output_path = "../output"

	arg.train_worker = 8
	arg.valid_worker = 4
	arg.test_worker = 4

	arg.wandb = False
	arg.wandb_project = "segmentation"
	arg.wandb_entity = "cv4"

	arg.custom_name = "resnest269e_panet_all"

	return arg