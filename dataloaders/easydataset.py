import os.path
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms


class easydataset(Dataset):
    def __init__(self, opt, transforms, split='train'):
        self.root = opt.dataset
        self.split = split
        self.path = os.path.join(self.root, split)

        self.images = []
        self.masks = []
        self.transforms = transforms
        self.masktransform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((opt.resize, opt.resize)),
            torchvision.transforms.ToTensor()
        ])

        # Get Mask List
        for _, _, filelist in os.walk(self.path):
            i = 0
            while i < len(filelist):
                if '.png' in filelist[i]:
                    self.masks.append(os.path.join(self.root, self.split, filelist[i]))
                    i += 1
                    continue
                if '.jpg' in filelist[i]:
                    self.images.append(os.path.join(self.root, self.split, filelist[i]))
                    i += 1
                    continue

        assert (len(self.images) == len(self.masks))

        print(f'Number of images in {split}: {len(self.images)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = Image.open(self.images[item]).convert('RGB')
        target = Image.open(self.masks[item])
        return self.transforms(img), self.masktransform(target)
