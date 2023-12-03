# Add model module to syspath
import sys, os
from os.path import dirname, abspath

model_dir = dirname(dirname(abspath(__file__))) 
sys.path.insert(0,model_dir)

from model.ConvModel import ConvNet
from model.classifer import ConvolutionalNeuralNet

import numpy as np

#  Instantiating model
conv_net = ConvNet()
model = ConvolutionalNeuralNet(conv_net,data_path='./data')
print(conv_net)

# Get classes and split images into train, test and validation
train_image_paths,test_image_paths,classes = model.select_images_data_sets()
# Create datasets
train_dataset,validation_dataset = model.create_datasets(train_image_paths,test_image_paths,classes)

# Get results from training 
results_dict = model.train_k_fold(train_dataset,validation_dataset,num_epochs=30,num_folds=5)

# Extract train losses and accuracy per epoch
train_losses_per_fold= [fold['train_loss_per_epoch'] for fold in results_dict]
train_acc_per_fold = [fold['train_accuracy_per_epoch'] for fold in results_dict]

# Create a list of train losses mean and train accuracy mean per epoch at each fold
train_losses = [np.mean(k) for k in zip(*train_losses_per_fold)]
train_acc = [np.mean(k) for k in zip(*train_acc_per_fold)]

# Extract validation losses and accuracy per epoch
val_losses_per_fold= [fold['val_loss_per_epoch'] for fold in results_dict]
val_acc_per_fold = [fold['val_accuracy_per_epoch'] for fold in results_dict]

# Create a list of validation losses mean and validation accuracy mean per epoch at each fold
val_losses = [np.mean(k) for k in zip(*val_losses_per_fold)]
val_acc = [np.mean(k) for k in zip(*val_acc_per_fold)]

# Plot losses
model.plot_loss_and_accuracy(train_losses,train_acc,val_losses,val_acc)

print("End")
