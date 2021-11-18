from easydict import EasyDict as eDict

def getArg():
	arg = eDict()

	arg.batch = 32
	arg.epoch = 10
	arg.lr = 1e-5
	arg.seed = 42
	arg.save_capacity = 5
	
	#arg.train_image_root = "../input/train2014"
	#arg.train_mask_root = "../input/train_mask"
	arg.train_image_root = "../input/val2014"
	arg.train_mask_root = "../input/val_mask"
	arg.val_image_root = "../input/val2014"
	arg.val_mask_root = "../input/val_mask"
	arg.output_path = "../output"

	arg.train_worker = 8
	arg.valid_worker = 8
	arg.test_worker = 8

	arg.wandb = True
	arg.wandb_project = "alchera"
	arg.wandb_entity = "cv4"

	arg.custom_name = "deeplabv3_3"
	
	arg.TTA = True
	arg.test_batch = 1

	return arg