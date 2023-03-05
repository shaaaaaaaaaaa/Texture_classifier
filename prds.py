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
df = pd.DataFrame(columns=['name','标注类别名称','类别编号'])
df.to_csv('myurban100.csv', mode='w', header=True,encoding='utf-8-sig')
print(model1)
model1.eval()
print(model2)
model2.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#print(model)
torch.no_grad()
print('over!!!')
paths = "/home/ljy/Jiays/DIV2K_train_HR_patches"
# paths = '/home/ljy/Jiays/test2/urban100'
# path = "/home/ljy/Jiays/test2/tes"
model2.cuda()
model1.cuda

names = os.listdir(paths)
# print(names)
resu_list = []
for name in names:
    image_path = os.path.join(paths, name)
    print(image_path)


    img = Image.open(image_path).convert('RGB')
    # img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    tran = transforms.ToTensor()
    trans = transforms.Compose([
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    img = trans(img)
    img = torch.unsqueeze(img ,dim=0)
    print(img.shape)
    model1.cuda()
    model2.cuda()
    img = img.to(device)
    outputs = model1(img)  # outputs，out1修改为你的网络的输出
    print(outputs.shape)
    out = list(model2.children())[1][6](outputs)
    
    print(out.shape)
    out = out.to(device)
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
    print(int(pred[0]))
    # degre = int(index[0])
    # print(degre)
    lists = ["banded","blotchy","braided","bubbly","bumpy","chequered","cobwebbed","cracked","crosshatched","crystalline","dotted","fibrous","flecked","freckled","frilly","gauzy","grid","grooved","honeycombed","interlaced","knitted","lacelike","lined","marbled","matted","meshed","paisley","perforated","pitted","pleated","polka-dotted","porous","potholed","scaly","smeared","spiralled","sprinkled","stained","stratified","striped","studded","swirly","veined","waffled","woven","wrinkled","zigzagged"]
    
    print(lists[pred])
    df = pd.DataFrame(columns=['name','标注类别名称','类别编号'], data=[[name,lists[pred],int(pred[0])]])
    df.to_csv('div2k.csv', mode='a', header=False)
ns = np.array(resu_list)
print(ns.shape)
nss = ns.squeeze( 1)
print(nss.shape)
np.save("div2k.npy",nss)
