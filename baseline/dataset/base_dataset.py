from torch.utils.data import Dataset
import cv2
import os


category_names = [
    'Background', 'Body', 'RightHand', 'LeftHand', 'LeftFeet', 'RightFeet', 
    'RightThigh', 'LeftThigh', 'RightCalf', 'LeftCalf', 'LeftArm', 'RightArm', 
    'LeftForeArm', 'RightForeArm','Head'
    ]

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

class CustomDataset(Dataset):
    """COCO format"""
    image_names = []
    num_classes = 15
    def __init__(self, image_root, mask_root=None, mode='train', transform=None):
        super().__init__()
        self.mode = mode
        self.image_root = image_root # original images
        self.mask_root = mask_root
        self.transform = transform
        self.setup()

    def setup(self):
        """
        saves path of each images
        """
        self.image_names = os.listdir(self.image_root)

    def __getitem__(self, index: int):
        
        images = cv2.imread(os.path.join(self.image_root, self.image_names[index]))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB) #FIXME albu issue

        if (self.mode in ('train', 'val')):
            # imagename에서 확장자 떼고 .png 붙이기
            file_name = os.path.splitext(self.image_names[index])[0]
            file_name += ".png"
            masks = cv2.imread(os.path.join(self.mask_root, file_name))

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.image_names)
