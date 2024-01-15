# Emotion Recognition Model

![Web APP](/CV01.png)
![input](/CV02.png)
![output](/CV03.png)

This repository contains a deep learning model for emotion recognition using facial images. The model is built using TensorFlow and Keras and is trained to classify faces into two emotion categories: "Sad Person" and "Happy Person."

## Data
FER Dataset was used. The dataset contains 28,709 images of faces with seven emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The dataset was preprocessed to extract only the "Sad" and "Happy" emotion categories. The dataset was split into training and validation sets with a 70:30 ratio. The training set contains 20,096 images, and the validation set contains 8,613 images.

## Model Architecture

The emotion recognition model consists of a convolutional neural network (CNN) with a series of convolutional, pooling, normalization, and fully connected layers. Below is a detailed description of the model architecture:

![Layers](/output.png)

### Input Layer
- **Type:** Conv2D
- **Activation:** ReLU
- **Padding:** Same
- **Kernel Size:** (3, 3)
- **Input Shape:** (48, 48, 1)

### Hidden Layers
1. **Conv2D Layer:**
   - Filters: 64
   - Activation: ReLU
   - Padding: Same
   - Kernel Size: (3, 3)
   - Batch Normalization
   - Max Pooling: (2, 2)
   - Dropout: 0.2

2. **Conv2D Layer:**
   - Filters: 128
   - Activation: ReLU
   - Padding: Same
   - Kernel Size: (3, 3)
   - Batch Normalization
   - Max Pooling: (2, 2)
   - Dropout: 0.2

3. **Conv2D Layer:**
   - Filters: 256
   - Activation: ReLU
   - Padding: Same
   - Kernel Size: (3, 3)
   - Batch Normalization
   - Max Pooling: (2, 2)
   - Dropout: 0.2

4. **Conv2D Layer:**
   - Filters: 512
   - Activation: ReLU
   - Padding: Same
   - Kernel Size: (3, 3)
   - Batch Normalization
   - Max Pooling: (2, 2)
   - Dropout: 0.2

5. **Conv2D Layer:**
   - Filters: 1024
   - Activation: ReLU
   - Padding: Same
   - Kernel Size: (3, 3)
   - Batch Normalization
   - Max Pooling: (2, 2)
   - Dropout: 0.2

### Fully Connected Layers
1. **Dense Layer:**
   - Neurons: 64
   - Activation: ReLU
   - Batch Normalization
   - Dropout: 0.2

2. **Dense Layer:**
   - Neurons: 128
   - Activation: ReLU
   - Batch Normalization
   - Dropout: 0.2

3. **Dense Layer:**
   - Neurons: 256
   - Activation: ReLU
   - Batch Normalization
   - Dropout: 0.2

4. **Dense Layer:**
   - Neurons: 512
   - Activation: ReLU
   - Batch Normalization
   - Dropout: 0.2

### Output Layer
- **Type:** Dense
- **Activation:** Sigmoid
- **Neurons:** 1 (Binary Classification)

### Model Compilation
- **Optimizer:** Adam (learning rate: 0.0001)
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy

### Training
- **Data Augmentation:**
  - Rotation Range: 10 degrees
  - Width Shift Range: 0.05
  - Height Shift Range: 0.05
  - Shear Range: 0.5
  - Zoom Range: 0.05
  - Horizontal Flip: True
  - Rescaling: 1/255

- **Early Stopping:**
  - Monitor: Validation Loss
  - Patience: 25 epochs

### Model Visualization
The model architecture can be visualized using the `visualkeras` library. The layered view provides an intuitive representation of the neural network structure.

### Prediction
![Prediction](/model_4.png)

#### Note: Ensure that you have the necessary libraries and dependencies installed before running the code. Refer to the provided code snippets and adjust the file paths accordingly.
