import torch
from torchvision import transforms
import torch.nn.functional as F
from encoding.models.deepten import DeepTen
from PIL import Image

# Load pre_trained Deepten model
nclass = 47
model = DeepTen(nclass,'resnet50')
model.load_state_dict(torch.load('model4.pt',map_location='cpu'),False)
model.eval()
# Load and preprocess input image
image_path = '/home/tangb_lab/cse30011373/jiays/dataSet/dtd/images/banded/banded_0139.jpg'
input_image = Image.open(image_path).convert('RGB')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
input_tensor = preprocess(input_image)
input_batch = torch.unsqueeze(input_tensor ,dim=0)

# run input batch
with torch.no_grad():
    output = model(input_batch)
# get predicted class probabilities
probabilities = F.softmax(output[0],dim=0)

texture_list = ["banded","blotchy","braided","bubbly","bumpy","chequered","cobwebbed","cracked","crosshatched","crystalline","dotted","fibrous","flecked","freckled","frilly","gauzy","grid","grooved","honeycombed","interlaced","knitted","lacelike","lined","marbled","matted","meshed","paisley","perforated","pitted","pleated","polka-dotted","porous","potholed","scaly","smeared","spiralled","sprinkled","stained","stratified","striped","studded","swirly","veined","waffled","woven","wrinkled","zigzagged"]
print(probabilities)
top_prob, top_catid = torch.topk(probabilities,5)
print(top_catid)
for i in range(5):
    print(texture_list[top_catid[i]])
