# import libraries

import time
import torch
from torch import optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
from pandas.core.common import flatten
import copy
import numpy as np
import random
import glob
from tqdm import tqdm,trange
from time import sleep

from mean_std_loader import StatsFromDataSet
import torch.nn as nn
import torch.nn.functional as F


#######################################################
#               Define Dataset Class
#######################################################

class ManOrWomankDataset(Dataset):
    """
        Arguments:
            image_paths (string): Path to the csv file with annotations.
            transform (string, optional): Directory with all the images.
            classes (callable): Optional transform to be applied
                on a sample.
    """
    def __init__(self, image_paths, classes,transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.classes = classes
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        #image = cv2.imread(image_filepath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = read_image(image_filepath)
        #######################################################
        #      Create dictionary for class indexes
        #######################################################
        idx_to_class =  {i:j for i, j in enumerate(self.classes)}
        class_to_idx =  {value:key for key,value in idx_to_class.items()}
        
        # Replace split char by '/' on unix systems
        label = image_filepath.split('\\')[-2]
        label = class_to_idx[label]
        if self.transform:
            image = self.transform(image)
        
        return image, label

#######################################################
#               Define Neural Net with Convolution
#######################################################

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # 3 input image channel, 6 output channels, 
	    # 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 53 * 53 , 120) # 16 out channels * 53 * 53 from image dimension after two convolutions on 224x224 images
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #print("Input shape = ", x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        #print("Shape after conv1 = ", x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        #print("Shape after conv2 = ", x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #print("Shape after flatten = ", x.shape)
        x = F.relu(self.fc1(x)) # relu is an activation function
        #print("Shape after fc1 = ", x.shape)
        x = F.relu(self.fc2(x))
        #print("Shape after fc2 = ", x.shape)
        x = self.fc3(x)
        #print("Final shape = ", x.shape)
        return x

# 2. Define Transforms

#######################################################
#               Define Transforms
#######################################################


#To define an augmentation pipeline, you need to create an instance of the Compose class.
#As an argument to the Compose class, you need to pass a list of augmentations you want to apply. 
#A call to Compose will return a transform function that will perform image augmentation.
#(https://albumentations.ai/docs/getting_started/image_augmentation/)
root_path = '.\\data'
train_data_path = root_path + '\\train' 
test_data_path = root_path + '\\test'

#Resize images to 50 x 50

img_width = 224
img_height = 224
batch_size = 16


def get_train_test_tranfomers(w,h,mean_train,std_train,mean_test,std_test):
    train_transforms_pt = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((w,h)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean_train,std_train),
        #lambda x: np.moveaxis(x.numpy(), 0, 3)
    ])
    test_transforms_pt = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((w,h)),
        transforms.ToTensor(),
        transforms.Normalize(mean_test,std_test),
        #lambda x: np.moveaxis(x.numpy(), 0, 3)
    ])
    return train_transforms_pt, test_transforms_pt




####################################################
#       Create Train, Valid and Test sets
####################################################

# Get the mean and std from a folder of images
# We use a custom helper class fo find

train_image_paths = [] #to store image paths in list
classes = [] #to store class values

#1.
# get all the paths from train_data_path and append image paths and class to to respective lists
# eg. train path-> 'images/train/26.Pont_du_Gard/4321ee6695c23c7b.jpg'
# eg. class -> 26.Pont_du_Gard
for data_path in glob.glob(train_data_path + '/*'):
    classes.append(data_path.split('\\')[-1]) 
    train_image_paths.append(glob.glob(data_path + '/*'))
    #1.1. Once images are extracted, we calculate the mean and std per folder/per class
    #1.2. Use the calculated mean and std to call a train transformer per class and apply 
    #1.3. Call a test transformer per class and p
    
train_image_paths = list(flatten(train_image_paths))
random.shuffle(train_image_paths)

print('train_image_path example: ', train_image_paths[0])
print('class example: ', classes[0])

#2. Split train/validation from train image paths (80,20)

# A colon on the left side of an index means everything before, but not including, the index; All the data up to the beginning of 80% of the train set, which is 80%
# A colon on the right side of an index means everything after the specified index : All the data after 80% of the data, which is 20%
train_image_paths, valid_image_paths = train_image_paths[:int(0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):] 

#3.# Create the test_image_paths based on test data path
test_image_paths = []
for data_path in glob.glob(test_data_path + '\*'):
    test_image_paths.append(glob.glob(data_path + '\*'))

test_image_paths = list(flatten(test_image_paths))

print("Train size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths), len(test_image_paths)))


#######################################################################
#                 Crete Tranformators for train and test folder
#######################################################################

train_stats = StatsFromDataSet(data_path=train_data_path,batch_size=batch_size)
train_normalized_loader = train_stats.init_large_loader(w=img_width,h=img_height)
mean_train_set,std_train_set = train_stats.batch_mean_and_std(train_normalized_loader)

test_stats = StatsFromDataSet(data_path=test_data_path,batch_size=batch_size)
test_normalized_loader = test_stats.init_large_loader(w=img_width,h=img_height)
mean_test_set,std_test_set = test_stats.batch_mean_and_std(test_normalized_loader)

# Get transformers using the mean and the std from original dataset
train_tranforms, test_transforms = get_train_test_tranfomers(img_width,img_height,mean_train=mean_train_set.tolist(),std_train=std_train_set.tolist(),
                                                            mean_test=mean_test_set,std_test=std_test_set)

# 3.3. Get datasets and apply transformers


   
#######################################################
#                  Create Dataset
#######################################################

train_dataset = ManOrWomankDataset(train_image_paths,classes,transform=train_tranforms)
test_dataset = ManOrWomankDataset(test_image_paths,classes,transform=test_transforms)
valid_dataset = ManOrWomankDataset(valid_image_paths,classes,transform=test_transforms) #test transforms are applied

print('The shape of tensor for 50th image in train dataset: ',train_dataset[49][0].shape)
print('The label for 50th image in train dataset: ',train_dataset[49][1])



#######################################################
#                  Visualize Dataset
#         Images are plotted after augmentation
#######################################################

def visualize_augmentations(dataset, idx=0, samples=10, cols=5, random_img = False,classes=classes):
    
    dataset = copy.deepcopy(dataset)
    #we remove the normalize and tensor conversion from our augmentation pipeline
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    idx_to_class =  {i:j for i, j in enumerate(classes)}

    
        
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    for i in range(samples):
        if random_img:
            idx = np.random.randint(1,len(train_image_paths))
        image, lab = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
        ax.ravel()[i].set_title(idx_to_class[lab])
    plt.tight_layout(pad=1)
    plt.show()    

#visualize_augmentations(train_dataset,np.random.randint(1,len(train_image_paths)), random_img = True)


# 6. Creating the DataLoader

#######################################################
#                  Define Dataloaders
#######################################################

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=True
)


test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

#batch of image tensor
print(next(iter(train_loader))[0].shape)

#batch of the corresponding labels
print(next(iter(train_loader))[1].shape)

class_names = test_loader.dataset.classes
print('Class names:', class_names)

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 60
plt.rcParams.update({'font.size': 20})

def imageshow(input, title):
    # torch.Tensor => numpy
    input = input.numpy().transpose((1, 2, 0))
    # undo image normalization
    mean = np.array(mean_train_set.tolist())
    std = np.array(std_train_set.tolist())
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # display images
    plt.imshow(input)
    plt.title(title)
    plt.show()

# load a batch of train image
iterator = iter(train_loader)

# visualize a batch of train image
inputs, classes = next(iterator)
out = torchvision.utils.make_grid(inputs[:4])
#imageshow(out, title=[class_names[x] for x in classes[:4]])





if torch.cuda.is_available():
  device = torch.device('cuda:0')
  print('Running on the GPU')
else:
  device = torch.device('cpu')
  print('Running on the CPU')

# Defining accuracy function for test dataset
def accuracy(model,test_loader:DataLoader,criterion,epoch):

    running_loss = 0.
    running_corrects = 0
    test_dataset_size = len(test_loader.dataset)
    
    with torch.no_grad():
        for i,data in enumerate(test_loader):
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction            
            _, predictions = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() + labels.size(0)
            running_corrects += torch.sum(predictions == labels.data)

            if i == 0:
                print('[Prediction Result Examples]')
                #images = torchvision.utils.make_grid(inputs[:4])
                #imshow(images.cpu(), title=[class_names[x] for x in labels[:4]])
                #images = torchvision.utils.make_grid(inputs[4:8])
                #imshow(images.cpu(), title=[class_names[x] for x in labels[4:8]])
        # compute the accuracy over all test images
        epoch_loss = running_loss / test_dataset_size
        epoch_acc = running_corrects / test_dataset_size * 100.
        print('[Validation #{}] Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
    return epoch_acc


#  defining accuracy function
def accuracy2(network, dataloader):
    network.eval()
    total_correct = 0
    total_instances = 0
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)
        predictions = torch.argmax(network(images), dim=1)
        correct_predictions = sum(predictions==labels).item()
        total_correct+=correct_predictions
        total_instances+=len(images)
    return round(total_correct/total_instances, 3)

# Define savemode function
def save_model():
    PATH = './men_woman.pth'
    torch.save(model, PATH)

# Define train model

def train(model:Net,train_loader:DataLoader,test_loader:DataLoader,num_epochs:int=20):

    #Optmization
    optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()
    best_accuracy = 0.0

    #  creating log
    log_dict = {
        'training_loss_per_batch': [],
        'validation_loss_per_batch': [],
        'training_accuracy_per_epoch': [],
        'validation_accuracy_per_epoch': []
    }

    losses = []
    epochs = []
    running_loss = 0.0

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(f' Epoch {epoch+1}/{num_epochs} ')
        train_losses = []

        num_samples = len(train_loader)
    
        # Setting tqdm progress bar
        border = "=" * 50
        clear_border = "\r" + " " *len(border) + "\r"
        
        for i, data in enumerate(train_loader, 1):
            sleep(0.01)
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data
            #  Resetting gradients
            optimizer.zero_grad() 

            # Get predictions
            predictions = model(images)
            # Computing loss
            loss = loss_function(predictions, labels) 
            # Add loss value to log (val 2)
            ################ (DELETE)
            log_dict['training_loss_per_batch'].append(loss.item())
            train_losses.append(loss.item())
            ################ (DELETE)
            #  computing gradients
            loss.backward()
            #  updating weights
            optimizer.step()
            #Store training data
            epochs.append(epoch + i / num_samples)
            losses.append(loss.item())    
        
            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
            if i % 100 == 99:    
                # print every 1000 (twice per epoch) 
                #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                print(clear_border + '[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                print(border)
                # zero the loss
                running_loss = 0.0
        
        #################### Training second code ##########################
        with torch.no_grad():
            print('deriving training accuracy...')
            #  computing training accuracy
            train_accuracy = accuracy2(model, train_loader)
            log_dict['training_accuracy_per_epoch'].append(train_accuracy)
            print(f'training accuracy: {train_accuracy}')

        #################### Training second code ##########################

        
        """ Validation Phase """
        
        # Compute and print the average accuracy fo this epoch whever thn tested over all 10000 test images
        model.eval()
        accuracy_test_val = accuracy(model,test_loader,loss_function,epoch)
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy_test_val))
        

       
        ###################### (DELETE - VALIDATION 2)########

        # Validation 2
        #  setting convnet to evaluation mode

        print('validating 2 way...')
        val_losses = []

        with torch.no_grad():
            for images, labels in test_loader:
                #  sending data to device
                images, labels = images.to(device), labels.to(device)
                #  making predictions
                predictions = model(images)
                #  computing loss
                val_loss = loss_function(predictions, labels)
                log_dict['validation_loss_per_batch'].append(val_loss.item())
                val_losses.append(val_loss.item())

            #  computing accuracy
            print('deriving validation accuracy...')
            accuracy_test_2 = accuracy2(model, test_loader)
            log_dict['validation_accuracy_per_epoch'].append(accuracy_test_2)
            print(f'validation accuracy: {accuracy_test_2}')

        train_losses  = np.array(train_losses).mean()
        val_losses   = np.array(val_losses).mean()

        print(f'training_loss: {round(train_losses , 4)}  training_accuracy: '+
        f'{train_accuracy}  validation_loss: {round(train_losses , 4)} '+  
        f'validation_accuracy: {accuracy_test_2}\n')
        
        
        ###################### (DELETE)################################################
        ###############################################################################
            
        # we want to save the model if the accuracy is the best
        if accuracy_test_val > best_accuracy:
            save_model()
            best_accuracy = accuracy_test_val
        

    print('Finished Training')

    

    return np.array(epochs),np.array(losses),log_dict


# Test model
# Function to show the images
def images_show(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function to test the model with a batch of images and show the labels predictions
def testBatch(model):
    model.eval()
    start_time = time.time()

    with torch.no_grad():
        running_loss = 0.
        running_corrects = 0

        for i, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if i == 0:
                print('[Prediction Result Examples]')
                images = torchvision.utils.make_grid(inputs[:4])
                imageshow(images.cpu(), title=[class_names[x] for x in labels[:4]])
                images = torchvision.utils.make_grid(inputs[4:8])
                imageshow(images.cpu(), title=[class_names[x] for x in labels[4:8]])

        epoch_loss = running_loss / len(valid_loader.dataset)
        epoch_acc = running_corrects / len(valid_loader.dataset) * 100.
        print('[Validation #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() - start_time))

if __name__ == "__main__":
    
    # Let's build our model
    #train(5)
    model = Net()
    print(model)
    num_epochs = 3
    epochs_list, losses_list,log = train(model, train_loader=train_loader,test_loader=test_loader,num_epochs=num_epochs)

    # Test which classes performed well
    #accuracy_test(model)

    # Test with batch of images
    testBatch(model)

