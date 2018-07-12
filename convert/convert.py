import sys
from os.path import dirname, realpath
sys.path.insert(1, dirname(dirname(realpath(__file__))))

from model import Yolonet, Mobilenetv2_base
import torch
import torch.nn as nn

def weights_normal_init(model):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)


basenet = Mobilenetv2_base()
yolonet = Yolonet(basenet)

weights_normal_init(yolonet)

y = torch.load('mobilenetv2_1.pth')
x = yolonet.state_dict()

i = 0
for k in list(x.keys()):
    if 'basenet' in k:
        x[k] = y['module.' + k[k.find('.')+1:]]
        i += 1

print(i)
torch.save(x, 'yolo_mobilenet_init.pth')
