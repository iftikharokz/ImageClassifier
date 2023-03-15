import torch
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import nn
from torch import optim
import argparse

# cd ImageClassifier
# python train.py flowers --epochs

def main():
        
    arg = getCommandLineArgs()
    data_dir = arg.data_dir
    epochs = int(arg.epochs)
    lr = float(arg.learning_rate)
    modelName = arg.arch
    modelDirectory = arg.save_dir
    hidden = int(arg.hidden_units)
    addGPU = arg.gpu
    print("data_dir",data_dir,"epochs",epochs,"lr",lr,"modelName",modelName,"hidden",hidden,"addGPU",addGPU)
    if modelName == "vgg16":
        arch = models.vgg16(pretrained=True)
    elif modelName == "vgg13":
         arch = models.vgg13(pretrained=True)
            
    if addGPU and torch.cuda.is_available():
        device = torch.device('cuda') 
        print("GPU ifti")
    else:
         device = torch.device('cpu') 
         print("cpu ifti")
            
    data,train_data = getData(data_dir)
    
    final_output_size = len(data['training'].dataset.classes)
    
    modle = model_def(arch,hidden,device,final_output_size)
    trained_mdl,criterion = train_model(epochs,data,device,modle,lr)
    test_trained_model(trained_mdl,data,device,criterion)
    save_Model(epochs,modle,arch,modelDirectory,lr,hidden,train_data,final_output_size)
    
def getCommandLineArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir",help = "This will be the data set directory/path and its must to be provided")
    parser.add_argument("--learning_rate",help = "Lr like 0.001 etc",default = 0.001)
    parser.add_argument("--epochs",help = "Model iteration",default = 5)
    parser.add_argument("--arch",help = "model name like vgg16",default = "vgg13")
    parser.add_argument("--save_dir",help = "loaction where the checkpoint will be saved",default="checkpoint.pth")
    parser.add_argument("--hidden_units",help = "hidden layer neuron count",default = 4096)
    parser.add_argument("--gpu",action='store_true',help = "Add gpu support")
    prc = parser.parse_args()
    return prc 


def getData(data:str):
    
    train_dir = data + '/train'
    valid_dir = data + '/valid'
    test_dir =  data + '/test'
    
    data_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
     ])
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224,0.225])
        ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
          ])
    load_data_sets = {
                    "train":datasets.ImageFolder(train_dir,transform=data_transforms),
                    "test":datasets.ImageFolder(test_dir,transform=test_transforms),
                    "validation":datasets.ImageFolder(valid_dir,transform=valid_transforms) 
    }
    data = {
        "training":torch.utils.data.DataLoader(load_data_sets["train"],batch_size=64,shuffle=True),
        "validation":torch.utils.data.DataLoader(load_data_sets["validation"],batch_size=64,shuffle=True),
        "testing":torch.utils.data.DataLoader(load_data_sets["test"],batch_size=64,shuffle=True)
    }
    return data ,load_data_sets


def model_def(model,hidden,device,output):
    
    for p in model.parameters():
        p.requires_grad = False
    in_features = 25088
    classifier = nn.Sequential(
                             nn.Linear(in_features,hidden),
                             nn.ReLU(),
                             nn.Dropout(p=0.35),
                             nn.Linear(hidden,2048),
                             nn.ReLU(),
                             nn.Dropout(p=0.3),
                             nn.Linear(2048, output),
                             nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier
#     print(model,"///////////////////")
    model.to(device)
    return model

def train_model(epochs,data,device,model,learning_rate):
    steps = 0
    running_loss = 0
    print_every = 30
    criterion = nn.NLLLoss()
    model.to(device)
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        for inputs, labels in data["training"]:
            steps+=1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            forwardPro = model.forward(inputs)
            loss = criterion(forwardPro, labels)
            loss.backward()
            optimizer.step()
#             print("_____________________________________________________",steps)
            running_loss += loss.item()
            if steps % print_every == 0:
              valid_loss = 0
              accuracy = 0
              model.eval()
              with torch.no_grad():
                    for inputs, labels in data["validation"]:
                        inputs, labels = inputs.to(device), labels.to(device)
                        # Forward pass
                        forwardPro = model.forward(inputs)
                        loss = criterion(forwardPro, labels)
                        valid_loss += loss.item()

                        ps = torch.exp(forwardPro)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
              print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valition loss: {valid_loss/len(data['validation']):.3f}.. "
                      f"Vadation accuracy: {accuracy/len(data['validation']):.3f}")
              running_loss = 0
    return model , criterion


def test_trained_model(model,data,device,criterion):
    model.eval()
    test_loss = 0
    accuracy = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data["testing"]:

            inputs, labels = inputs.to(device), labels.to(device)
            forwardPro = model.forward(inputs)
            batch_loss = criterion(forwardPro, labels)
            test_loss += batch_loss.item()

            ps = torch.exp(forwardPro)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.sum(equals).item()
            total += labels.size(0)

    avg_test_loss = test_loss / len(data["testing"])
    avg_accuracy = accuracy / total

    print(f"Test loss: {avg_test_loss:.3f}.. "
          f"Test accuracy: {avg_accuracy:.3f}")
    
    
def save_Model(epochs,trained_model,arch,save_loc,lr,hidden,train_data,final_output_size):
    trained_model.class_to_idx = train_data['train'].class_to_idx
    checkpoint = {
                 'arch': arch,
                 'in_feature':25088,
                 'hidden':hidden,
                  'output':final_output_size,
                 'learning_rate':lr,
                 'class_to_idx': trained_model.class_to_idx,
                 'state_dict': trained_model.state_dict(),
                 'epochs': epochs
             }
    torch.save(checkpoint,save_loc)
    print("medel saved successfully")
        
if __name__ == '__main__':
          main()
          

    