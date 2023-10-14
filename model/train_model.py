from ConvModel import ConvNet
from classifer import ConvolutionalNeuralNet
import numpy as np
if __name__ == "__main__":
    
    #  Instantiating model
    conv_net = ConvNet()
    model = ConvolutionalNeuralNet(conv_net,data_path='../data')
    print(conv_net)
    
    # Get classes and split images into train, test and validation
    train_image_paths,test_image_paths,classes = model.select_images_data_sets(extra_test_set=False)
    # Create datasets
    train_dataset,test_dataset = model.create_datasets(train_image_paths,test_image_paths,classes)
    # Create dataloaders
    #train_dataloader,test_dataloader = model.create_data_loader(train_dataset,test_dataset)
    # Train model
    #log_dict = model.train(train_loader=train_dataloader,test_loader=test_dataloader,num_epochs=10,model_save_path='./best_gender_model.pth')
    log_dict = model.train_k_fold(train_dataset,test_dataset,num_epochs=30,num_folds=5)
    
    #Extract results for train and val loss and accuracy
    train_losses_per_fold= [fold['train_loss_per_epoch'] for fold in log_dict]
    train_acc_per_fold = [fold['train_accuracy_per_epoch'] for fold in log_dict]

    train_losses = [np.mean(k) for k in zip(*train_losses_per_fold)]
    train_acc = [np.mean(k) for k in zip(*train_acc_per_fold)]

    val_losses_per_fold= [fold['val_loss_per_epoch'] for fold in log_dict]
    val_acc_per_fold = [fold['val_accuracy_per_epoch'] for fold in log_dict]

    val_losses = [np.mean(k) for k in zip(*val_losses_per_fold)]
    val_acc = [np.mean(k) for k in zip(*val_acc_per_fold)]

    # Plot losses
    model.plot_loss_and_accuracy(train_losses,train_acc,val_losses,val_acc)
    
    print("End")
