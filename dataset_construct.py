import torch
from PIL import Image
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """
    自定义数据集
        因为数据（sample）是图片，不能一次性全部加载到内存，所以采用第二种方式，
        将其保存的路径加载到内存中，待需要的时候在进行读取。

        对于标签，如果比较简单，则可直接加载到内存
    """

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images = images_path
        self.label = images_class
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        if img.mode != "RGB":
            raise ValueError("Img is not RGB type")
        if self.transform is not None:
            img = self.transform(img)

        label = self.label[index]

        return img, label

    def __len__(self):
        return len(self.label)

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
