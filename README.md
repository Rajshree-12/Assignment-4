# Assignment-4
Neural Network using Keras dataset
Observations by making changes in the code related to various parameters:
1. Keeping just a single hidden layer with number of neurons in it much lesser than the input nodes, with the activation function being Relu for this layer and Softmax for output layer, we found that train accuracy as well as the test accuracy was comparatively lesser than
when the number of nodes for this hidden layer is around 900. Further increasing the number of nodes in this layer to around 1400, with all the other cases being the same gave a much better result. It is observed that although the train accuracy is slightly reduced but test accuracy is increased largely reaching upto 98.33%.
Then after for the rest cases being same as we increased the no. of epochs, it led to the case of overfitting with training accuracy being largely improved but a good amount of decrease has been observed in the test accuracy.
So we can say that keeping a large number of input nodes for first hidden layer is good cause in this case we have large number of input layer nodes, but better to keep it below twice the number of input layer neurons.Also the number of epochs should not be very large as it may lead to overfitting.
Also keeping more than one hidden layers has not yielded much improvement unless they all have same number of neurons for each layer.
Also unnecessary addition of it is just increasing the complexity and the computation time.
Also when I changed the optimizer function to 'SGD', the performance got reduced. So for the given multiple class classification the 'ADAM', optimizer is giving the best result when applied with appropriate no. of layers and neurons. And I also tried it by changing all the activation functions to sigmoid , but it again reduced the performance.
So here the Relu for hidden layers and softmax for output classication is best suited.
