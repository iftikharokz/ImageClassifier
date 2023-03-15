import torch
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import seaborn as sb
import argparse
import predict
from train import model_def
import json
import sys

# 

def getCommandLineArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir",help = "This will be the single image directory/path but you must have to give it") #
    parser.add_argument("--checkpoint",help = "checkpoint path",default='checkpoint.pth')
    parser.add_argument("--top_k",help = "top classes number",default = 5)
    parser.add_argument("--category_names",help = "Classes file",default = "cat_to_name.json")
    parser.add_argument("--gpu",action='store_true',help = "Add gpu support")
    prc = parser.parse_args()
    return prc 

def loadModel(filepath):
    
    args = getCommandLineArgs()
    gpu = args.gpu
    if  gpu and torch.cuda.is_available():
        device=torch.device('cuda')                    
    else:
        
        device=torch.device('cpu')
        print("The model saved in the checkpoint is trained on GPU can't be load on CPU")
        sys.exit(1)
    
    loadedMod = torch.load(filepath)
    model = model_def(loadedMod['arch'],loadedMod['hidden'],device,loadedMod['output'])
    model.class_to_idx = loadedMod['class_to_idx'] 
    model.load_state_dict(loadedMod['state_dict'])
    for p in model.parameters():
        p.requires_grad = False
    print("model loaded")
    return model
                    
def process_image(image_path,device):
    immage = Image.open(image_path)
    adjust = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    image = adjust(immage)
    
    return image.to(device)

def predict(image_path, model, topk,device,category_names):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with open(category_names, 'r') as f:
         cat_to_name = json.load(f)
    
    processed_image = process_image(image_path,device).to(device)
    processed_image.unsqueeze_(0)
    probs = torch.exp(model.forward(processed_image)).cpu()
    top_probs, top_labs = probs.topk(int(topk))

    idx_to_class = {}
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key

    np_top_labs = top_labs[0].numpy()

    top_labels = []
    for label in np_top_labs:
        top_labels.append(int(idx_to_class[label]))

    top_flowers = [cat_to_name[str(lab)] for lab in top_labels]
    
    return top_probs,  top_flowers 
def main():
    arg = getCommandLineArgs()
    image_dir = arg.image_dir
    checkpoint_path= arg.checkpoint
    top_k = arg.top_k
    category_names = arg.category_names
    gpu = arg.gpu
    if  gpu and torch.cuda.is_available():
        device=torch.device('cuda') 
        print("Cuda is active")
    else:
        device=torch.device('cpu')
        print("cpu is active")     
    
    model = loadModel(checkpoint_path)
    top_p, top_flowers, = predict(image_dir, model, top_k,device,category_names)
    print("....",top_p.tolist()[0])
    for i in range(int(top_k)):
        print("The Flower Named: {fname} probability is {prob}".format(fname=top_flowers[i], prob=top_p.tolist()[0][i]*100))

              
if __name__ == '__main__':
          main()