import os, glob
from PIL import Image
import models
import params
import torch
from torchvision import transforms
from torch.autograd import Variable
import dataset

weight = '/home/huyvd/Documents/works/crnn-pytorch-master/expr/netCRNN_0_200.pth'
input_size = (params.imgW, params.imgH)

nclass = len(params.alphabet) + 1


class PredictCRNN:
    def __init__(self):
        self.model = models.CRNN(params.imgH, params.nc, nclass, params.nh)
        self.model.load_state_dict(torch.load(weight))
        self.transformer = dataset.resizeNormalize((320, 32))
        self.model.eval()

    def predict(self, PILimg):
        img_transformed = self.transformer(PILimg)
        img_transformed = img_transformed.view(1, *img_transformed.size())
        input_img = Variable(img_transformed)
        preds = self.model(input_img)

        return preds


img_path = '/home/huyvd/Documents/works/test_crnn'
img_arr = glob.glob(os.path.join(img_path, '*'))

model = PredictCRNN()
for img in img_arr:
    pred_img = Image.open(img).convert('L')
    pred = model.predict(pred_img)
    print(f"pred: {pred}")







