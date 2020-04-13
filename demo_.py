import pandas as pd
import tokenizers
det = 'D:/data/iqiyi/personai_icartoonface_dettrain/icartoonface_dettrain.csv'


df = pd.read_csv(det, header=None , index_col=0, names=['x1', 'x2' , 'y1', 'y2'])

dfbox = df.loc['personai_icartoonface_dettrain_00001.jpg',:]
box = []
if issubclass(pd.Series,type(dfbox)):
    box = [list(dfbox.values)]
else:
    for i in range(len(dfbox)):
        x1, x2 , y1, y2 = dfbox.iloc[i,:]
        box.append([x1, x2, y1, y2])


for i in range(len(box)):
    print(box[i])


