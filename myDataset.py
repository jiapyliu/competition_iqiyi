import torch.utils.data
import os
from PIL import Image
import pandas as pd
from torchvision import transforms

class IcartoonDataset(torch.utils.data.Dataset):
    def __init__(self, root = 'D:\data\iqiyi\personai_icartoonface_dettrain', transforms=None):
        self.root = root
        self.transforms = transforms
        # 下载所有图像文件，为其排序
        # 确保它们对齐
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'icartoonface_dettrain'))))
        self.df = pd.read_csv(os.path.join(root, 'icartoonface_dettrain.csv'), header=None, index_col=0, names=['x1', 'x2' , 'y1', 'y2'])

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, 'icartoonface_dettrain',  self.imgs[idx])

        img = Image.open(img_path).convert("RGB")
        dfbox = self.df.loc[self.imgs[idx], :]
        box = []
        if issubclass(pd.Series, type(dfbox)):
            box = [list(dfbox.values)]
        else:
            for i in range(len(dfbox)):
                x1, x2, y1, y2 = dfbox.iloc[i, :]
                box.append([x1, x2, y1, y2])

        box = torch.as_tensor(box, dtype=torch.float32)
        if self.transforms is not None:
            mytransform = transforms.Compose([transforms.ToTensor()])
            img = mytransform(img)

        target = {}
        target['boxes'] = box
        target['labels'] = torch.zeros((len(box),), dtype=torch.int64)


        return img, target

    def __len__(self):
        return len(self.imgs)


def test():
    data = IcartoonDataset(transforms=True)

    first = data[6]
    img = first[0]
    img = transforms.ToPILImage()(img).convert('RGB')

    box = first[1]['boxes'].numpy()
    #
    def draw_det_rectangle(img,box):
        from PIL import Image, ImageDraw
        draw = ImageDraw.Draw(img)
        for i in range(len(box)):
            draw.rectangle(box[i])
        img.show()
    draw_det_rectangle(img, box)

# test()