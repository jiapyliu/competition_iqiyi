import pandas as pd

det = 'D:/data/iqiyi/personai_icartoonface_dettrain/icartoonface_dettrain.csv'
rec = 'D:/data/iqiyi/personai_icartoonface_rectrain/icartoonface_rectrain_det.txt'

det = pd.read_csv(det, header=None)
rec = pd.read_csv(rec, sep='\t', header=None)

print(len(det))
print(len(rec))

def draw_det_rectangle(num=5):
    from PIL import Image, ImageDraw
    df = det.head(num)
    for i in range(num):
        x1, y1, x2, y2 = df.iloc[i, :][1:]
        p = 'D:/data/iqiyi/personai_icartoonface_dettrain/icartoonface_dettrain/' + df.iloc[i, :][0]
        im = Image.open(p)
        draw = ImageDraw.Draw(im)
        draw.rectangle((x1, y1, x2, y2))
        print('x:{},y:{}'.format(x2 - x1, y2 - y1))
        im.show()


def draw_rec_rectangle(num=5):
    from PIL import Image, ImageDraw
    df = rec.head(num)
    for i in range(num):
        x1, y1, x2, y2 = df.iloc[i, :][1:]
        p = 'D:/data/iqiyi/personai_icartoonface_rectrain/icartoonface_rectrain/' + df.iloc[i, :][0]
        im = Image.open(p)
        draw = ImageDraw.Draw(im)
        draw.rectangle((x1, y1, x2, y2))
        print('x:{},y:{}'.format(x2 - x1, y2 - y1))
        im.show()


draw_det_rectangle()