# Neural Style Transfer:

**Neural style transfer** is a popular optimization technique which allows us to blend two images, a content image and a style reference image, and output a new image having the content features of the first image and the styling of the latter.

* It proves that it is possible to separate the style representation and content representations from a Convolutional Neural Network.
* It uses a pre-trained convolutional neural network to transfer styles from one image to another.
* A loss-function for content-image is defined which tries to minimize the difference between the features that are activated for the content-image and for the mixed-image, at different available layers in the network. 
* A loss function for style-image is defined for minimizing the mean-squared distance between the entries of the Gram matrix from the style image and the Gram matrix of the image generated after every iteration. The gram matrix provides us the correlation matrix between the filters.
* We then calculate the total loss by a defined linear combination of the content loss and the style loss according to our needs.
* Then Tensorflow is used for deriving the gradients of these loss functions easily. 
* I used the VGG19 CNN Model for getting the content and the style layers of the input images.
* VGG19 consists of 19 layers out of which 16 are convolutional layers and 3 are fully connected layers. The various layers of this model build-up to form the complete image. It is used for the classification of the images. Style transfer uses the pre-trained model to get the style and the content layers for the input images.
* Used the Block 5 layer for content and Block 4 for the style feature layers.
* Using the Tensorflow Keras API to import the model.
* The inputs images to this VGG19 Network are required to be processed before using them in the model. It converts the RGB to BGR format and adds some biases. In the end, we’ll have to de-process the image to see it correctly. 
* In the end, back-propagation to be used for changing the target image for reducing the losses. 
# Content Loss = ∑(new_image_content - base_image_content)²/2
# Style Loss = ∑(new_image_style_gram_matrix - base_image_style_gram_matrix)²/4*N^2*M^2	
