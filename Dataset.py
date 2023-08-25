from torch.utils.data import Dataset
from torchvision.io import read_image

class GenderDataset(Dataset):
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