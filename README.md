# MNIST Digit Classification Project

This project implements a neural network to classify handwritten digits using the MNIST dataset.

## Project Structure

- `data_loader.py`: Handles loading and preprocessing of the MNIST dataset.
- `imports.py`: Contains all necessary import statements.
- `neural_net.py`: Defines the neural network architecture.
- `train.py`: Contains the training loop and logic.
- `test.py`: Main script to run the model and perform digit classification.

## Installation

Ensure you have Python 3 installed. Install the required libraries:

pip install numpy torch pandas matplotlib

## Usage

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Run the main script: python test.py
4. The script will load the trained model and perform digit classification on the MNIST test set.

## Data

This project uses the MNIST dataset, which is typically included in TensorFlow and will be automatically downloaded when running the script for the first time.

## Model

The neural network architecture is defined in `neural_net.py`. You can modify the architecture or hyperparameters in this file to experiment with different configurations.

## Training

The training process is implemented in `train.py`. To retrain the model or adjust training parameters, modify this file and run it: python train.py

## Results

After running `test.py`, the script will output the model's accuracy on the MNIST test set. You may also see sample predictions and visualizations of the digits.

## Acknowledgements

- The MNIST dataset providers
- TensorFlow and other open-source libraries used in this project
