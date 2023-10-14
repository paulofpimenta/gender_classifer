# import libraries

import torch
from torch import optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader,ConcatDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from pandas.core.common import flatten
import copy
import numpy as np
import random
import glob
from tqdm.auto import tqdm
from statistics import mean
import torch.nn as nn
from sklearn.model_selection import KFold

import os
from PIL import Image
import torch.nn.functional as nnf

from mean_std_loader import StatsFromDataSet
from Dataset import GenderDataset
from ConvModel import ConvNet


#######################################################
#               Define Neural Net with Convolution
#######################################################

class ConvolutionalNeuralNet():

    #To define an augmentation pipeline, you need to create an instance of the Compose class.
    #As an argument to the Compose class, you need to pass a list of augmentations you want to apply. 
    #A call to Compose will return a transform function that will perform image augmentation.
    #(https://albumentations.ai/docs/getting_started/image_augmentation/)

    def __init__(self,network,img_width:int = 224, img_height:int = 224, data_path='data'):
      
        # Define parameters
        self.device = self.get_device()
        self.network = network.to(self.device)

        self.train_data_path = os.path.join(data_path,'train',"")
        self.test_data_path = os.path.join(data_path,'test',"")
        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = 32

    # 2. Define Transforms

    #######################################################
    #               Define Transforms
    #######################################################

    def get_train_test_tranfomers(self,w,h,mean_train,std_train,mean_test,std_test):
        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((w,h)),
            transforms.CenterCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomCrop(227),    
            transforms.ToTensor(),
            transforms.Normalize(mean_train,std_train),
            #lambda x: np.moveaxis(x.numpy(), 0, 3)
        ])

        train_transform_albu = A.Compose([
            A.SmallestMaxSize(max_size=160),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomCrop(height=128, width=128),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        test_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((w,h)),
            transforms.ToTensor(),
            transforms.Normalize(mean_test,std_test),
        ])
        return train_transforms, test_transforms


    def select_images_data_sets(self,extra_test_set=False):
        ####################################################
        #       Create Train, Valid and Test sets
        ####################################################

        # Get the mean and std from a folder of images
        # We use a custom helper class fo find

        train_image_paths = [] #to store image paths in list
        classes = [] #to store class values

        # Set working directory
        current_folder = os.path.dirname(os.path.abspath(__file__))
        os.chdir(current_folder)

        #1.
        # get all the paths from train_data_path and append image paths and class to to respective lists
        # eg. train path-> 'images/train/26.Pont_du_Gard/4321ee6695c23c7b.jpg'
        # eg. class -> 26.Pont_du_Gard
        for data_path in glob.glob(self.train_data_path + '/*'):
            # Replace the split by '\\' on windows 
            classes.append(data_path.split('/')[-1]) 
            train_image_paths.append(glob.glob(data_path + '/*'))
            #1.1. Once images are extracted, we calculate the mean and std per folder/per class
            #1.2. Use the calculated mean and std to call a train transformer per class and apply 
            #1.3. Call a test transformer per class and p
            
        train_image_paths = list(flatten(train_image_paths))
        random.shuffle(train_image_paths)

        print('\nTrain_image_path example: ', train_image_paths[0])
        print('Class example: ', classes[0], '\n')

        #2. Split train/validation from train image paths (80,20)

        # A colon on the left side of an index means everything before, but not including, the index; All the data up to the beginning of 80% of the train set, which is 80%
        # A colon on the right side of an index means everything after the specified index : All the data after 80% of the data, which is 20%
        #train_image_paths, test_image_paths = train_image_paths[:int(0.5*len(train_image_paths))], train_image_paths[int(0.5*len(train_image_paths)):] 

        #3.# Create the test_image_paths based on test data path
        test_image_paths = []
        for data_path in glob.glob(self.test_data_path + '/*'):
            test_image_paths.append(glob.glob(data_path + '/*'))

        test_image_paths = list(flatten(test_image_paths))
        random.shuffle(test_image_paths)

        # Create validation set from test dataset, if necessary
        if extra_test_set : 
            test_image_paths, valid_image_paths = test_image_paths[:int(0.8*len(test_image_paths))], test_image_paths[int(0.8*len(test_image_paths)):] 
            print("Train dataset size: {}\nTest dataset size: {}\nValidation dataset size: {}".format(len(train_image_paths), len(test_image_paths),len(valid_image_paths)))
            return train_image_paths,test_image_paths,valid_image_paths,classes
        else :
            print("Train dataset size: {}\nTest dataset size: {}".format(len(train_image_paths), len(test_image_paths)))
            return train_image_paths,test_image_paths,classes


    def __create_transformators_train_test(self):
    #######################################################################
    #                 Crete Tranformators for train and test folder
    #######################################################################

        train_stats = StatsFromDataSet(data_path=self.train_data_path,batch_size=self.batch_size)
        train_normalized_loader = train_stats.init_large_loader(w=self.img_width,h=self.img_height)
        mean_train_set,std_train_set = train_stats.batch_mean_and_std(train_normalized_loader)

        test_stats = StatsFromDataSet(data_path=self.test_data_path,batch_size=self.batch_size)
        test_normalized_loader = test_stats.init_large_loader(w=self.img_width,h=self.img_height)
        mean_test_set,std_test_set = test_stats.batch_mean_and_std(test_normalized_loader)

        # Get transformers using the mean and the std from original dataset
        train_tranforms, test_transforms = self.get_train_test_tranfomers(self.img_width,self.img_height,mean_train=mean_train_set.tolist(),std_train=std_train_set.tolist(),
                                                                    mean_test=mean_test_set,std_test=std_test_set)
        return train_tranforms,test_transforms


# 3.3. Get datasets and apply transformers

    def create_datasets(self,train_image_paths,test_image_paths,classes):
   
        #######################################################
        #                  Create Dataset
        #######################################################

        train_tranforms,test_transforms = self.__create_transformators_train_test()

        train_dataset = GenderDataset(train_image_paths,classes,transform=train_tranforms)
        test_dataset = GenderDataset(test_image_paths,classes,transform=test_transforms)
        #valid_dataset = GenderDataset(valid_image_paths,classes,transform=test_transforms) #test transforms are applied

        print('The shape of tensor for 50th image in train dataset: ',train_dataset[49][0].shape)
        print('The label for 50th image in train dataset: ',train_dataset[49][1])

        return train_dataset,test_dataset


    #######################################################
    #                  Visualize Dataset
    #         Images are plotted after augmentation
    #######################################################

    def visualize_augmentations(dataset, classes,train_image_paths,idx=0, samples=10, cols=5,random_img = False):
        
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


    def create_data_loader(self,train_dataset,test_dataset):
    #######################################################
    #                  Define Dataloaders
    #######################################################

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )     
        #valid_loader = DataLoader(
        #    valid_dataset, batch_size=self.batch_size, shuffle=True
        #)

        #batch of image tensor
        print(next(iter(train_loader))[0].shape)

        #batch of the corresponding labels
        print(next(iter(train_loader))[1].shape)

        class_names = test_loader.dataset.classes
        print('Class names:', class_names)

        return train_loader,test_loader

    def get_device(self):

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print('Running on the GPU using', torch.cuda.get_device_name(0))
        else:
            device = torch.device('cpu')
            print('Running on the CPU')
        return device

    # Define savemode function
    def save_model(self,path:str='./best_gender_model.pth'):
        torch.save(self.network.state_dict(),path)

    
    #  defining weight initialization function
    #def init_weights(self,module):
    #  if isinstance(module, nn.Conv2d):
    #    torch.nn.init.xavier_uniform_(module.weight)
    #    module.bias.data.fill_(0.01)
    #  elif isinstance(module, nn.Linear):
    #    torch.nn.init.xavier_uniform_(module.weight)
    #    module.bias.data.fill_(0.01)

    def train(self,dataloader, loss_fn, optimizer):
        
        # Sets the model in training mode.
        self.network.train()

        # Set epoch loss and correct values
        running_loss, running_corrects,train_epoch_corrects = 0, 0, 0

        # Iterate over the DataLoader for training data
        for batch_i, data in enumerate(dataloader, 0):

            self.network.to(torch.device(self.device))
            
            # Get inputs
            inputs, targets = data

            # Reshape input and target tensor accodingly with the device used
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Zero the gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # Compute prediction loss
                outputs = self.network(inputs)
                _,preds = torch.max(outputs,1)
                
                # Compute loss                
                loss = loss_fn(outputs, targets)
        
                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

            # write to logs
            running_loss += loss.item() * inputs.size(0)
            running_corrects += (preds== targets).sum()
            train_epoch_corrects += preds.size(0)

            # Print every 500 mini-batches
            #if batch_i % 500 == 499:
            #    current_loss = running_loss
            #    print('Loss after mini-batch %5d: %.3f' % (batch_i+ 1, current_loss / 500))
            #    current_loss = 0.0 # reset current loss to zero              
        
        #Training loss
        train_epoch_loss = (running_loss / train_epoch_corrects)
        
        # FTraining accuracy
        train_epoch_acc = (running_corrects.double() ) / train_epoch_corrects
        train_epoch_acc = train_epoch_acc.tolist()

        # return avg loss of epoch, acc of epoch
        return train_epoch_loss, train_epoch_acc
    

    def test(self,dataloader, loss_fn,optimizer):
        size = len(dataloader.dataset) # number of samples
        num_batches = len(dataloader) # batches per epoch

        self.network.eval() # Sets the model in test mode.
                
        # Set epoch loss and correct values
        epoch_loss, epoch_correct, num_samples = 0, 0, 0


        for i, data in enumerate(dataloader, 0):
            # Get inputs
            inputs, targets = data

            # Reshape input and target tensor accodingly with the device used
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            #
            optimizer.zero_grad()

            # No training for test data
            with torch.no_grad():
                # Compute prediction loss
                outputs = self.network(inputs)
                _,preds = torch.max(outputs,1)

                # Calculate loss
                loss = loss_fn(outputs, targets)

            # write to logs
            epoch_loss += loss.item() * inputs.size(0)
            epoch_correct += (preds == targets).sum()
            num_samples += preds.size(0)

        # Test loss
        test_loss = (epoch_loss / num_samples)
        
        # Test accuracy
        test_acc = (epoch_correct.double() ) / num_samples
        test_acc = test_acc.tolist(

        )
        # return avg loss of epoch, acc of epoch
        return test_loss, test_acc

    
    # Define train model
    def train_k_fold(self,train_dataset,test_dataset,num_epochs:int=10,num_folds:int=5):
        # Build test loader based on test dataset
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=self.batch_size)

        # Initialize the k-fold cross validation
        kf = KFold(n_splits=num_folds, shuffle=True)

        # Define loss function
        loss_function = nn.CrossEntropyLoss()

         # For fold results
        results = {}
        log_dict = []
        test_fold_losses = []
        test_fold_accs = []
        best_fold = 0
        best_accuracy = 0.0

        #concatenating Train/Test part; we split later.
        #full_dataset = ConcatDataset([train_dataset, test_dataset])
        

        # Loop through each fold
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset),1):

            fold_log = {
                'train_loss_per_epoch': [],
                'train_accuracy_per_epoch': [],
                'val_loss_per_epoch':[],
                'val_accuracy_per_epoch': [],
                'test_fold_losses' : [],
                'test_fold_acc' : [] 
            }

            print("---------------------------")
            print(f"Fold {fold}")
            print("---------------------------")      

            # Define the train and validation data loaders from train dataset for the current fold
            # Sample elements randomly from a given list of indexes
            train_loader = DataLoader (
                dataset=train_dataset,
                batch_size=self.batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(train_idx),
            )
            val_loader = DataLoader (
                dataset=train_dataset,
                batch_size=self.batch_size,
                sampler=torch.utils.data.SubsetRandomSampler(val_idx),
            )

            # Init network weights
            #self.network.apply(self.init_weights)

            # Initialize the model and optimizer
            optimizer = optim.SGD(self.network.parameters(), lr=0.0001, momentum=0.9)

            for epoch in range(1, num_epochs + 1):
                # Print epoch
                print(f"\nStart epoch {epoch}\n")

                # Get train and validation folds metrics
                train_loss, train_acc = self.train(train_loader,loss_function, optimizer)
                print('\t\t Training Epoch ({}) - Loss : {:.4f}, Acc {:.4f}'.format(epoch,train_loss,train_acc))

                val_loss, val_acc = self.test(val_loader, loss_function, optimizer)
                print('\t\t Validation({}) - Loss : {:.4f}, Acc : {:.4f}'.format(epoch,val_loss,val_acc))

                # Filling log
                fold_log['train_loss_per_epoch'].append(train_loss)
                fold_log['train_accuracy_per_epoch'].append(train_acc)
                fold_log['val_loss_per_epoch'].append(val_loss)
                fold_log['val_accuracy_per_epoch'].append(val_acc)

                #print('Training process has finished. Starting tos evauluate model...')

            #################################################################
            # TESTING FOLD WITH UNSEEN DATA               
            #################################################################
            
            test_loss, test_acc = self.test(test_loader, loss_function,optimizer)

            fold_log['test_fold_losses'] = test_loss
            fold_log['test_fold_acc'] = test_acc
    

            #Printing accuracy
            #tqdm.write(f'Accuracy for fold {fold} : {test_acc:.3f}')
            #print(f'Accuracy for fold {fold} : {test_acc:.3f}')
            print('\nLoss on fold({}) : {:.4f}, Acc : {:.4f}'.format(fold,test_loss,test_acc))

            results[fold] = test_acc

            # Append currrent fold results to log dictionary
            log_dict.append(fold_log)


            # We want to save the model if the accuracy is the best           
            # if test_accuracy > best_accuracy:
            #     print (f'Model\'s test accuracy ({test_accuracy}) on fold {fold + 1} is higher than best accuracy ({best_accuracy}). Saving model..')
            #     best_accuracy = test_accuracy
            #     best_fold = fold
            # else :
            #     print (f'Model\'s current accuracy ({test_accuracy}) on fold{fold + 1} is lower than the best accuracy {best_accuracy}. Model will not be saved')
        
        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {num_folds} FOLDS')
        print('--------------------------------')
        sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            sum += value
            print(f'Average: {sum/len(results.items())} %')
            
        self.save_model()

        return log_dict

    # Function to test the model with a batch of images and show the labels predictions
    def evaluate_dataset(self,valid_loader:DataLoader,classes,num_images_plot:int=None):
        self.network.eval()
        loss_function = nn.CrossEntropyLoss().to('cuda')

        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            running_loss = 0
            total_correct = 0
            total_instances = 0
            correct_predictions = 0

            for i, (inputs, labels) in enumerate(valid_loader):
                # Subsetting images and label
                inputs = inputs[:num_images_plot]
                labels = labels[:num_images_plot]

                # calculate outputs by running images through the network
                outputs = self.network(inputs)
                _,preds = torch.max(outputs, 1)
                loss = loss_function(outputs, labels)

                # Calculate loss and correct pred
                running_loss += loss.item() * inputs.size(0)
                correct_predictions = torch.sum(preds==labels).item()
                total_correct+=correct_predictions
                total_instances+=len(inputs)

                if i == 0:
                    # print labels
                    predicted_labels = [classes[preds[i]] for i in range(len(inputs))]
                    ground_true_labels = [classes[labels[i]] for i in range(len(inputs))]
                    print('Ground Truth: ', ''.join(f'{ground_true_labels}'))
                    print('Predicted: ', ''.join(f'{predicted_labels}'))
                    # Show images
                    selected_images = torchvision.utils.make_grid(inputs)
                    self.images_show(img=selected_images,title=predicted_labels)

            selected_images_acc = round(total_correct/total_instances, 3)

        print(f'Accuracy of the network on {len(valid_loader.dataset)} validation images: {selected_images_acc} %')
        
    
    def predict(self,image: Image):
        # Pre-process image & create a mini-batch as expected by the model
      
        preprocess = transforms.Compose([
                 transforms.Resize(224),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.6527503132820129, 0.48301947116851807, 0.4047924280166626], 
                                     std= [0.235576793551445, 0.20688192546367645, 0.19748619198799133]),
             ])
        # Preprocess and squeeze     
        input_tensor = preprocess(image)
        # Adding an extra dimension of depth 1 on first dimension
        input_batch = input_tensor.unsqueeze(0) 

        self.network.eval()
    
        with torch.no_grad():
            # convert input from list to Tensor
            X = torch.Tensor(input_batch)

            # move tensor to device
            X = X.to(self.device)

            # Get outputs (as logits)
            output = self.network(X)

            # Convert logits to softmax
            probability = nnf.softmax(output, dim=1)
            top_prob, top_class = probability.topk(1, dim = 1)
            # Get predictions
            _,prediction = torch.max(output, 1)
            classes=['female','male']
            predicted_class = classes[prediction.item()]
            #Saving probability of prediction
            pred_prob = top_prob.item()

            #Title and response
            response = {predicted_class.upper():str(pred_prob)}
            
        return response
                
        # Test model
    # Function to show the images
    def images_show(self,img,title):
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 60
        plt.rcParams.update({'font.size': 20})
        
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()

        plt.figure("Predictions")
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(title)
        plt.show(block=False)

    def plot_losses_and_accuracy(self,loss_train,loss_test):
        plt.figure("Train loss")
        plt.plot(loss_train,color='tab:blue')
        plt.xlabel('Loss')
        plt.ylabel('Batch')
        plt.legend(['Test'])
        plt.title('Train loss per batch')
        plt.show(block=True)

        plt.figure("Test loss")
        plt.plot(loss_test,color='tab:orange')
        plt.xlabel('Loss')
        plt.ylabel('Batch')
        plt.legend(['Test'])
        plt.title('Test loss per batch')
        plt.show(block=True)
    
    def plot_accuracy(self,train_acc,test_acc):
        plt.figure("Accuracy")
        # Folds
        epoch = list(range(1,len(train_acc) + 1,1))
        plt.plot(train_acc,'-o')
        plt.plot(test_acc,'-o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train','Test'])
        plt.title('Train vs Test Accuracy by Fold')
        plt.show(block=True)
    
    def plot_prediction(self,prediction,image):
        title = [*prediction.keys()][0].upper() + " => Probability: " + "{:,.2f} %".format(float([*prediction.values()][0]) * 100)
        # Plot
        plt.imshow(image)
        plt.title(title)
        plt.show(block=True)
    
    def plot_loss_and_accuracy(self,train_loss,train_acc,test_loss,test_acc):
        plt.figure("Accuracy and Loss KFold")
        epoch = list(range(1,len(train_loss) + 1))
        plt.plot(epoch,train_loss,'-o',label='Train loss')
        plt.plot(epoch,test_loss,'-o',label='Validation loss')
        plt.plot(epoch,train_acc,'-o',label='Train accuracy')
        plt.plot(epoch,test_acc,'-o',label='Validation accuracy')
        plt.title("Mean of 10-Fold Cross-Validation Accuracy and Loss of Train and Validation")
        plt.ylabel('Accuracy and loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show(block=True)