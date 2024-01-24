# Project: Follow-ME Project 
---
[image1]: ./images/networks.png
[image2]: ./images/train.PNG
[image3]: ./images/output.PNG
[image4]: ./images/output1.PNG
[image5]: ./images/output2.PNG
[image6]: ./images/score.PNG

## Network Architecture- Network is a Fully Convolutional Network 
---
### Three major parts of the network- Encoder, 1x1, and Decoder
### 1. Encoder: - 
#### This part of the network is responsible for detecting different features in the image. Each layer of the encoder is responsible for detecting certain types of features. For detecting an object, the 1st layer detects basic small straight lines, the 2nd layer detects curves and edges part of the image, and then the 3rd layer, it is detecting the object. 
##### The encoder has n no. of convolutional layers. The value of n depends upon the complexity of the problem and the size fo the data set.
##### The convolutional layer can be of a regular type or separable type.
##### Project architecture  has 2 layers of the Separable convolutional layer with kernel size 3x3, strides = 2, and padding = 'SAME' for each convolution. 
##### A separable convolution layer is used because:
###### a. Reduce runtime.
###### b. Efficient.
###### c. Useful for Mobile Applications.
###### d. Reduce Overfitting.
##### Relu activation function is applied to the output of each Separable convolutional layer.  The relu function is used to increase the non-linearity of the network 
##### Input to Encoder is an image of dimensions 160x160x3 
##### Separable convolution is applied to input with filter size = 32, then we get our 1st convolutional layer of dimension 80x80x32.
##### Separable convolution is applied to the 1st layer with filter size = 64, then we get our 2nd convolutional layer of dimension 40x40x64.
##### 2nd layer is then fed to 1x1 convolutional layer.
##### 32 and 64  filter sizes were chosen as per considering the size of the dataset. These sizes are not too complex or simple for the size of the dataset available( around 4000 images for training).
---
### 2. 1x1 convolutional layer:-
#### A 1x1 convolutional layer is nothing but a convolutional layer with stride and kernel size equal to  1x1.
#### A 1x1 convolutional layer is used in place of the fully connected neural network because a fully connected neural network loses special information,  while in the case of 1x1 special information is preserved. Further, this preserved special information is in the Decoder layer.
#### In the project filter size is 256 for 1x1 convolution. 
#### Output of 1x1 convolution layer is provided to the Decoder part.
---
### 3. Decoder:-
#### Decoder is nothing but transpose convolution. It is used to recreate a segmented image from the spatial and feature information provided by the previous layer.  
#### To recreate the same size segmented image that of input, we should use equal no. of encoder and decoder layers.
##### In the project we have 2 upsampling layers.
##### Each upsampling layer has bilinear upsampling, followed by skip connection and then 3 Separable convolution layers with strides = 1, kernel size  = 3x3, padding = 'SAME'.
##### The advantage of using a skip connection is that we can reproduce the more finely tuned segmented images. In the project, the skip connection is concatenated type because concatenated skip connection allows us to add two layers with different filter depths. But for using skip connections, we need to apply a few convolutional layers for converting the concatenated layer into a layer with the required no. of filter depth.
##### The 2 upsampling layers have filter depts as 64 and 32 respectively.
---
### Output layer:-
#### A regular convolutional layer with stride = 1, kernel = 1x1, and filter depth = several classes is used along with the softmax activation function.
##### Stride and kernel size is equal to one because we already have the image dimension equal to the dimension of the input image from the decoder part of the neural network. We are using the softmax function to normalize multiple numbers of classes and filter depth is used to covert the final output in the proper number of the classes(In the project, the number of output classes is equal to 3).  
![alt text][image1]
---

## Different architecture that can be used.
### The following modifications can be done to the network
#### 1. We can increase the number of encoder, decoder, and 1x1 layers to make our network more complex, but for the more complex network, we need a large amount of data for training. Maybe somewhere around 10,000 training images. For 4000 images the accuracy of this architecture is less than for the current one.  
#### 2. We can use a regular convolutional layer instead of a separable convolutional layer. But in this case, we need to train more number parameters, which requires more data.
#### 3. To increase the efficiency of training without overfitting for the same architecture, we can use more training data and dropouts between two layers. Dropouts prevent the network from getting overtrained to certain data.
#### 4. A pooling layer can also be added to reduce the computational power.
---
## 1x1 convolutions 
#### It is convlutions layer with kernel size = 1x1 and stride = 1.
#### It is  used to add complexity to your network. 
#### It acts as a mini Neural Network and is a nonlinear classifier.
#### It is used in Fully CNN instead of a fully connected layer because it stores features as well as special information of an input. It is present between the encoder and decoder layer of FCNN.
---
## Fully Connected    
#### It is a normal network of layers of neurons. 
#### Each neuron in a layer is connected to every neuron in an adjacent layer.
#### It acts as a linear and nonlinear classifier depending upon the number of neuron layers in the network.
#### It is generally used where we need to classify inputs.
#### EX:- Used in normal architectures like perceptron and even at the end part of CNN.
---
### Can this network be used for following another object?
#### Ya this network can be trained to follow another object, but for that, you will just need to produce a proper data set. The data provided for the project can't used for following another object.
#### In the case of developing  an architecture for classifying dogs and cars you may need an extra layer of encoders and decoders. This is because on the surface level dogs and cats are much alike. You need extra layers for detailed classification.
---
# Output 
## Training loss graph
![alt text][image2]
## Output1
![alt text][image3]
## Output2
![alt text][image4]
## Output3
![alt text][image5]
## Score 
![alt text][image6]


