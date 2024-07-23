from imports import * 
from hyperparameters import *   
# train_data = pd.read_csv(r'C:\Users\nihar\Documents\github\Deep_learning\MNIST\dataset\train\train.csv')
# test_data = pd.read_csv(r'C:\Users\nihar\Documents\github\Deep_learning\MNIST\dataset\test\test.csv')

# MNIST dataset
train_dataset = datasets.MNIST(root='../../data',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = datasets.MNIST(root='../../data',
                                            train=False, 
                                            transform=transforms.ToTensor())

# Data loader
train_loader = DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                                            batch_size=batch_size, 
                                            shuffle=False)