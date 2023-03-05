import torch, glob, cv2
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from deepten import DeepTen,DeepTen2
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

nclass = 47
model1 = DeepTen2(nclass)
model2 = DeepTen(nclass)
model1.load_state_dict(torch.load('model4.pt'),False)
model2.load_state_dict(torch.load('model4.pt'),False)
df = pd.DataFrame(columns=['name','标注类别名称'])
df.to_csv('my.csv', mode='w', header=True,encoding='utf-8-sig')
print(model2)

#print(model)
torch.no_grad()
print('over!!!')
paths = "/home/ljy/Jiays/DIV2K_train_HR_patches"
path = '/home/ljy/Jiays/test2/data'
# path = "/home/ljy/Jiays/test2/tes"
names = os.listdir(paths)
# print(names)
resu_list = []
for name in names:
    image_path = os.path.join(paths, name)
    print(image_path)


    img = Image.open(image_path).convert('RGB')
    # img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    tran = transforms.ToTensor()
    trans = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(128),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    
    img = trans(img)
    img = torch.unsqueeze(img ,dim=0)
    img = img.to(device)
    print(img.shape)
    outputs = model1(img)  # outputs，out1修改为你的网络的输出
    print(outputs.shape)
    out = list(model2.children())[1][6](outputs)
    print(out.shape)
    print('over')
    # x = out.cpu().detach().numpy()
    x = outputs.tolist()
    print(type(x))
    print(len(resu_list))
    resu_list.append(x)
    # print(type(x))
    print('========')
    print(out)
    pred = out.max(1,keepdim=True)[1] # 获取预测结果

    # predicted,index = out.max(1)
    print('------')
    print(pred)
    # degre = int(index[0])
    # print(degre)
    lists = ["banded","blotchy","braided","bubbly","bumpy","chequered","cobwebbed","cracked","crosshatched","crystalline","dotted","fibrous","flecked","freckled","frilly","gauzy","grid","grooved","honeycombed","interlaced","knitted","lacelike","lined","marbled","matted","meshed","paisley","perforated","pitted","pleated","polka-dotted","porous","potholed","scaly","smeared","spiralled","sprinkled","stained","stratified","striped","studded","swirly","veined","waffled","woven","wrinkled","zigzagged"]
    
    print(lists[pred])
    df = pd.DataFrame(columns=['name','标注类别名称'], data=[[name,lists[pred]]])
    df.to_csv('my.csv', mode='a', header=False)
ns = np.array(resu_list)
print(ns.shape)
nss = ns.squeeze( 1)
print(nss.shape)
np.save("result.npy",nss)
