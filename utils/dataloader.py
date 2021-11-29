import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


class CamoObjDataset(data.Dataset):
    def __init__(self, image_root, gtmask_root, gtedge_root, pseudomask_root, gray_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gtmasks = [gtmask_root + f for f in os.listdir(gtmask_root) if f.endswith('.png')]
        self.gtedges = [gtedge_root + f for f in os.listdir(gtedge_root) if f.endswith('.png')]
        self.pseudomasks = [pseudomask_root + f for f in os.listdir(pseudomask_root) if f.endswith('.png')]
        self.grays = [gray_root + f for f in os.listdir(gray_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.gtmasks = sorted(self.gtmasks)
        self.gtedges = sorted(self.gtedges)
        self.pseudomasks = sorted(self.pseudomasks)
        self.grays = sorted(self.grays)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gtmask_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.gtmask_rgb_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.gtedge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.pseudomask_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.gray_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gtmask = self.binary_loader(self.gtmasks[index])
        gtmask_rgb = self.rgb_loader(self.gtmasks[index])
        gtedge = self.rgb_loader(self.gtedges[index])
        pseudomask = self.rgb_loader(self.pseudomasks[index])
        gray = self.binary_loader(self.grays[index])
        image = self.img_transform(image)
        gtmask = self.gtmask_transform(gtmask)
        gtmask_rgb = self.gtmask_rgb_transform(gtmask_rgb)
        gtedge = self.gtedge_transform(gtedge)
        pseudomask = self.pseudomask_transform(pseudomask)
        gray = self.gray_transform(gray)

        return image, gtmask, gtmask_rgb, gtedge, pseudomask, gray, index

    def filter_files(self):
        assert len(self.images) == len(self.gtmasks)
        assert len(self.images) == len(self.gtedges)
        assert len(self.images) == len(self.pseudomasks)
        assert len(self.images) == len(self.grays)

        images = []
        gtmasks = []
        gtedges = []
        pseudomasks = []
        grays = []
        for img_path, gtmask_path, gtedge_path, pseudomask_path, gray_path in zip(self.images, self.gtmasks, self.gtedges, self.pseudomasks, self.grays):
            img = Image.open(img_path)
            gtmask = Image.open(gtmask_path)
            gtedge = Image.open(gtedge_path)
            pseudomask = Image.open(pseudomask_path)
            gray = Image.open(gray_path)
            if img.size == gtmask.size:
                images.append(img_path)
                gtmasks.append(gtmask_path)
                gtedges.append(gtedge_path)
                pseudomasks.append(pseudomask_path)
                grays.append(gray_path)
        self.images = images
        self.gtmasks = gtmasks
        self.gtedges = gtedges
        self.pseudomasks = pseudomasks
        self.grays = grays

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gtmask_root, gtedge_root, pseudomask_root, gray_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):

    dataset = CamoObjDataset(image_root, gtmask_root, gtedge_root, pseudomask_root, gray_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader, dataset.size

class test_dataset:
    def __init__(self, image_root, pseudomask_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.pseudomasks = [pseudomask_root + f for f in os.listdir(pseudomask_root) if f.endswith('.png')]
        self.images = sorted(self.images)
        self.pseudomasks = sorted(self.pseudomasks)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.pseudomask_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        pseudomask = self.rgb_loader(self.pseudomasks[self.index])
        HH = image.size[0]
        WW = image.size[1]
        image = self.transform(image).unsqueeze(0)
        pseudomask = self.pseudomask_transform(pseudomask).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, pseudomask, HH, WW, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


