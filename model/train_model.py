import numpy as np
from model.conv_model import ConvNet
from model.conv_neural_net import ConvolutionalNeuralNet
from torchsummary import summary

# 1. Instantiate model
conv_net = ConvNet()
model = ConvolutionalNeuralNet(conv_net, data_path="./data")

# 2. Print model summary
summary(conv_net, (3, model.img_height, model.img_width), model.batch_size)

# 3. Get classes and split images into train, test and validation
train_image_paths, test_image_paths, classes = model.select_images_data_sets()

# 4. Create datasets 
train_dataset, validation_dataset = model.create_datasets(train_image_paths, test_image_paths, classes)

# 5. Get results from training
results_dict = model.train_k_fold(train_dataset, validation_dataset, num_epochs=30, num_folds=5)

# 6. Extract train losses and accuracy per epoch
train_losses_per_fold = [fold["train_loss_per_epoch"] for fold in results_dict]
train_acc_per_fold = [fold["train_accuracy_per_epoch"] for fold in results_dict]

# 7. Create a list of train losses mean and train accuracy mean per epoch at each fold
train_losses = [np.mean(k) for k in zip(*train_losses_per_fold)]
train_acc = [np.mean(k) for k in zip(*train_acc_per_fold)]

# 8. Extract validation losses and accuracy per epoch
val_losses_per_fold = [fold["val_loss_per_epoch"] for fold in results_dict]
val_acc_per_fold = [fold["val_accuracy_per_epoch"] for fold in results_dict]

# 9. Create a list of validation losses mean and validation accuracy mean per eposch at each fold
val_losses = [np.mean(k) for k in zip(*val_losses_per_fold)]
val_acc = [np.mean(k) for k in zip(*val_acc_per_fold)]

# 10.Plot losses
model.plot_loss_and_accuracy(train_losses, train_acc, val_losses, val_acc)

print("End")
