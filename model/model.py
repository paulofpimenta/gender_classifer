from classifer import ConvolutionalNeuralNet
from ConvModel import ConvNet

if __name__ == "__main__":
    
    #  Instantiating model
    conv_net = ConvNet()
    model = ConvolutionalNeuralNet(conv_net)
    print(conv_net)
    
    # Get classes and split images into train, test and validation
    train_image_paths,test_image_paths,valid_image_paths,classes = model.select_images_data_sets()
    # Create datasets
    train_dataset,test_dataset,valid_dataset = model.create_datasets(train_image_paths,test_image_paths,valid_image_paths,classes)
    # Init network weights
    model.network.apply(model.init_weights)
    # Create dataloaders
    train_dataloader,test_dataloader,valid_dataloader = model.create_data_loader(train_dataset,test_dataset,valid_dataset)
    # Train model
    log_dict = model.train(train_loader=train_dataloader,test_loader=test_dataloader,num_epochs=10,model_save_path='./best_gender_model.pth')
    # Plot losses
    model.plot_losses(log_dict['training_loss_per_batch'],log_dict['test_loss_per_batch'])
    # Plot accuracy
    model.plot_accuracy(log_dict['training_accuracy_per_epoch'],log_dict['test_accuracy_per_epoch'])
    # Let us look at how the network performs on the whole validation dataset.
    model.evaluate_dataset(valid_dataloader,classes,7)
    # Show predictions on random images

    print("End")
