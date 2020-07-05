
import pretrainedmodels
import sys
sys.path.append("../src/")
from model_dispatcher import MODEL_DISPATCHER

if __name__ =="__main__":
    model_pretrained = pretrainedmodels.__dict__['resnet34'](pretrained=None)
    print(model_pretrained)
    print("X"*500)
    model_not_pretrained = MODEL_DISPATCHER["resnet34"](pretrained=False)
    print(model_not_pretrained)