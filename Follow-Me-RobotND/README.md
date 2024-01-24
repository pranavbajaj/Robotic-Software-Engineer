#Project: Follow-ME Project 
---
[image1]: ./images/networks.png
[image2]: ./images/train.PNG
[image3]: ./images/output.PNG
[image4]: ./images/output1.PNG
[image5]: ./images/output2.PNG
[image6]: ./images/score.PNG

##Network Architecture- Network is Fully Convolutional Network 
---
###Three major part of the network- Encoder, 1x1, and Decoder
###1.Encoder: - 
####This part of the network is responsible for detecting different features in the image. Each layer of the encoder is responsible for detecting certain types of features. Like for detecting an object, the 1st layer is detecting basic small straight lines, 2nd layer is detecting curves and edges part of the image and then in the 3rd layer, it is detecting the object. 
#####The encoder has n no. of convolutional layers. The value of n depends upon the complexity of problem and size fo data set.
#####Convolutional layer can be of a regular type or separable type.
##### Project architecture  have 2 layers of the Separable convolutional layer with kernel size 3x3, strides = 2 and padding = 'SAME' for each convolution. 
##### Separable convolution layer is used because:
######a. Reduce runtime.
######b. Efficient.
######c. Useful for Mobile Application.
######d. Reduce Overfitting.
#####Relu activation function is applied to the output of each Separable convolutional layer.  Relu function is used to increase the non-linearity of the network 
#####-Input to Encoder is an image of dimensions 160x160x3 
#####-Separable convolution is applied to input with filter size = 32, then we get our 1st convolutional layer of dimension 80x80x32.
#####-Separable convolution is applied to the 1st layer with filter size = 64, then we get our 2nd convolutional layer of dimension 40x40x64.
#####-2nd layer is then fed to 1x1 convolutional layer.
##### 32 and 64  filter sizes were chosen as per considering the size of the dataset. These sizes are not too complex or simple for the size of the dataset available( around 4000 images for training).
---
###2. 1x1 convolutional layer:-
#### 1x1 convolutional layer is nothing but a convolutional layer with stride and kernel size equal to  1x1.
#### 1x1 convolutional layer is used in place of the fully connected neural network because in a fully connected neural network loses special information,  while in case of 1x1 special information is preserved. Further, this preserved special information is in Decoder layer.
####In the project filter size is 256 for 1x1 convolution. 
####Output of 1x1 convolution layer is provided to Decoder part.
---
###3. Decoder:-
#### Decoder is nothing but transpose convolution. It is used to recreate a segmented image from the spacial and feature information provided by the previous layer.  
#### To recreate the same size segmented image that of input, we should use equal no. of encoder and decoder layers.
#####In the project we have 2 upsampling layers.
##### Each upsampling layer has bilinear upsampling, followed by skip connection and then 3 Separable convolution layers with strides = 1, kernel size  = 3x3, padding = 'SAME'.
#####Adventage of using skip connection is that we are able to reproduce more finely tuned segmented image. In the project, skip connection is concatenated type because concatenated skip connection allows us to add two layers with different filter depths. But for using skip connections, after that, we need to apply a few convolutional layers for converting the concatenated layer into a layer with required no. of filter depth.
#####The 2 upsampling layers has filter depts as 64 and 32 respectively.
---
###Output layer :-
####A regular convolutional layer with stride = 1, kernel = 1x1 and filter depth = number of classes is used along with softmax activation function.
#####Stride and kernel size is equal to one because we already have the image dimension equal to the dimension of the input image form decoder part of the neural network. We are using softmax function to normalize multiple numbers of classes and filter depth is used to covert the final output in the proper number of the classes(In the project, the number of output classes is equal to 3).  
![alt text][image1]
---

## Different architecture that can be used.
###Following modifications can be done to the network:-4
####1. We can increase the number of encoder, decoder, and 1x1 layers to make our network more complex, but for the more complex network, we need a large amount of data for training. Maybe somewhere around 10,000 training images. For 4000 images the accuracy of this architecture is less than for the current one.  
####2. We can use regular convolutional layer instead of separable convolutional layer. But in this case, we need to train more number of parameters, which requires more data.
####3. To increase the efficiency of training without overfitting for the same architecture, we can use more number of training data and dropouts between two layer. Dropouts prevent the network from getting overtrained to certain data.
####4. Pooling layer can also be added to reduce the computational power.
---
##Hyperparameters
###I was able to reach a good tunned hyperparameters after performing the following numbers of different iterations.:-

######1st
######With filter depth as :- Encoder - 64, 128, 256, 1x1 - 256, Decoder - 128, 64, num_classes
######learning_rate = 0.001
######batch_size = 64
######num_epochs = 10
######steps_per_epoch = 200
######validation_steps = 50
######workers = 2
######final score = 0.2049

######2nd 
######With filter depth as :- Encoder - 64, 128, 256, 1x1 - 256, Decoder - 128, 64, num_classes
######learning_rate = 0.01
######batch_size = 100
######num_epochs = 10
######steps_per_epoch = 200
######validation_steps = 50
######workers = 2
######final score = 0.3601

######3rd 
######With filter depth as :- Encoder - 32, 64, 1x1 - 64, Decoder - 32, num_classes
######learning_rate = 0.01
######batch_size = 100
######num_epochs = 10
######steps_per_epoch = 40
######validation_steps = 50
######workers = 2
######final score = 9.05793933083e-08

######4rd 
######With filter depth as :- Encoder - 32, 64, 1x1 - 64, Decoder - 32, num_classes
######learning_rate = 0.01
######batch_size = 40
######num_epochs = 10
######steps_per_epoch = 200
######validation_steps = 50
######workers = 2
######final score = 0.225

######5rd 
######With filter depth as :- Encoder - 32, 64, 1x1 - 64, Decoder - 32, num_classes
######learning_rate = 0.001
######batch_size = 100
######num_epochs = 10
######steps_per_epoch = 200
######validation_steps = 50
######workers = 2
######final score = 0.0

######6rd 
######With filter depth as :- Encoder - 32, 64, 1x1 - 128, Decoder - 64, 32
######learning_rate = 0.001
######batch_size = 100
######num_epochs = 10
######steps_per_epoch = 200
######validation_steps = 50
######workers = 2
######final score = 0.3985


######7rd 
######With filter depth as :- Encoder - 32, 64, 1x1 - 256, Decoder - 64, 32
######learning_rate = 0.01
######batch_size = 40
######num_epochs = 15
######steps_per_epoch = 200
######validation_steps = 50
######workers = 2
######final score = 0.312

######8rd 
######With filter depth as :- Encoder - 32, 64, 1x1 - 256, Decoder - 64, 32
######learning_rate = 0.01
######batch_size = 40
######num_epochs = 12
######steps_per_epoch = 200
######validation_steps = 50
######workers = 2
######final score = 0.40959

######9rd 
######With filter depth as :- Encoder - 32, 64, 1x1 - 256, Decoder - 64, 32(with one seperable convolution at end)
######learning_rate = 0.01
######batch_size = 40
######num_epochs = 12
######steps_per_epoch = 200
######validation_steps = 50
######workers = 2
######final score = 0.35

######10rd 
######With filter depth as :- Encoder - 32, 64, 1x1 - 256, Decoder - 64, 32(with one seperable convolution at end)
######learning_rate = 0.001
######batch_size = 100
######num_epochs = 10
######steps_per_epoch = 200
######validation_steps = 50
######workers = 2
######final score = 0.355

######11th
######With filter depth as :- Encoder - 32, 64, 1x1 - 256, Decoder - 64,32
######learning_rate = 0.01
######batch_size = 40
######num_epochs = 12
######steps_per_epoch = 200
######validation_steps = 50
######workers = 2
######final score = 0.43924
---
##1x1 convolutions 
####It is convlutions layer with kernel size = 1x1 and stride = 1.
####It is  used to add complexity to your network. 
####It acts as a mini Neural Network and is non linear classifier.
####It is used in Fully CNN instead of fully connected layer because it stores feature as well as special information of an input. It is present between encoder and decoder layer of FCNN.
---
##Fully Connected    
####It is a normal network of layer of neurons. 
####Each neuron in a layer is connected to every neuron in adjesent layer.
####It acts a linear and non linear classifier depending upon the number of neuron layers in the network.
####It is generally used at a place where we just need to classify inputs.
####EX:- Used in normal architectures like perceptron and even at the end part of CNN.
---
###Can this network be used for following another object?
####Ya this network can be trained to follow another object, but for that you will just need produce proper data set. The data provided for the project can't used for following another objects.
####In case of developing  a architecture for classfifying dog and car you may need extra layer of encoders and decoders. This is because on surface level dog and cat are much alike. You need extra layers for delailed classification.
---
#Output 
##Training loss graph
![alt text][image2]
##Output1
![alt text][image3]
##Output2
![alt text][image4]
##Output3
![alt text][image5]
##Score 
![alt text][image6]


