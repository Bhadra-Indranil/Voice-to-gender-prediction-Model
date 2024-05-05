# Voice to gender prediction

Welcome to the project! This repository contains a custom implementation of a neural network for classification tasks using Python and NumPy. The project includes data preprocessing, model training, evaluation, and predictions.

## Components

1. Common voice dataset
2. MFCC (Mel-frequency cepstrum coefficients)
3. Neural Network model for binary classification

## Dataset

Common Voice is a corpus of speech data read by users on the Common Voice website (http://voice.mozilla.org/), and based upon text from a number of public domain sources like user submitted blog posts, old books, movies, and other public speech corpora. Its primary purpose is to enable the training and testing of automatic speech recognition (ASR) systems.

## Usage

### Data prepartion

Ensure your data is in the correct format, with features in a CSV file where each row represents an instance and the last column is the target label. Preprocess the data appropriately to feed it into the neural network.

### Model Initialization

Initialize the neural network by specifying the input size, hidden layer sizes, output size, learning rate (eta), and dropout probability (p). These parameters define the structure and behavior of your neural network.

### Training Model

Train the model by iterating over the dataset multiple times (epochs). During each epoch, the model performs forward propagation to compute predictions and backpropagation to adjust the weights and biases to minimize the loss.

### Evaluating the model

After training, evaluate the model's performance using accuracy and other metrics such as precision, recall, and F1 score. This helps in understanding how well the model performs on unseen data.

## Model Architecture

Model Consist of differnt layers

1. Input layer
2. Hidden layers
3. Output layer
4. Cross entropy loss

### Training the Model

**Forward Propagation** for computing the activation of each layers

**Backpropagation** for calculating gradients and adjusting the weights and biases to minimize the cross-entropy loss.

**Gradient Descent** for updating the model parameters using the computed gradients to reduce the loss over time.

### Function explanation

**Parameters**

1.X_train: Training feature data.

2.y_train: Training target data.

3.X_val: Validation feature data.

4.y_val: Validation target data.

5.epochs: The maximum number of epochs to run the training process.

6.patience: The number of epochs to wait for an improvement in validation accuracy before stopping early.

**Training loop**

- For each epoch

1. Forward pass calculate the predicted Output for the given Input and updates the weights using the Backpropagation technique.

2. **Accuracy calculations** :

- Compute the training accuracy by comparing the predicted training output with the actual output
- Compute the validation output by comparing the predicted validation output and actual validation output

3. **Early stopping** :

- If the validation accuracy improves, update _best_acc_ and reset _patience_counter_ to 0. Save the current model.
- If the validation accuracy does not improve, increment _patience_counter_.
- If _patience_counter_ exceeds the patience threshold, stop the training early.

4. After training, load the model with the best validation accuracy.

## Evaluation

The model's performance is primarily evaluated using accuracy. Our model achieves an impressive accuracy of approximately **95%**, indicating that it correctly classifies a high proportion of the test instances.

## Results

This neural network achieves accuracy of **~95%** . This high accuracy shows that the model performs well on the given classification task

## Contributing

We welcome contributions to improve the model and its performance. If you'd like to contribute, please fork the repository, make your changes, and submit a pull request. We appreciate your contributions and will review them promptly.

## License

[MIT](https://choosealicense.com/licenses/mit/)

This project is licensed under the MIT License, allowing you to freely use, modify, and distribute the software.
