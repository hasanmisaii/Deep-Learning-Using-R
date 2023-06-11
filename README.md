# [Deep Learning Methods](https://arxiv.org/abs/2106.11342)
Deep learning methods refer to a set of algorithms and techniques used in the field of artificial intelligence and machine learning. These methods are designed to model and learn complex patterns and representations from large amounts of data using neural networks with multiple layers.

Some popular deep learning methods include:

1. Convolutional Neural Networks (CNNs): CNNs are widely used for image recognition and computer vision tasks. They are designed to automatically learn spatial hierarchies of features from images by using convolutional layers, pooling layers, and fully connected layers.

2. Recurrent Neural Networks (RNNs): RNNs are used for sequence data processing tasks such as natural language processing and speech recognition. They have recurrent connections that allow them to remember information from previous time steps, making them suitable for tasks with sequential dependencies.

3. Long Short-Term Memory (LSTM): LSTM is a type of RNN architecture that addresses the vanishing gradient problem. It has memory cells and gates that can selectively remember or forget information over long sequences, making it effective for capturing long-term dependencies.

4. Generative Adversarial Networks (GANs): GANs are used for generative modeling tasks such as image generation. They consist of a generator network that generates samples and a discriminator network that tries to distinguish between real and generated samples. The two networks are trained in a competitive manner, pushing each other to improve their performance.

5. Autoencoders: Autoencoders are used for unsupervised learning and data compression tasks. They consist of an encoder network that maps input data to a lower-dimensional latent space and a decoder network that reconstructs the input data from the latent space representation. Autoencoders can learn meaningful representations and capture important features of the input data.

These are just a few examples of deep learning methods, and there are many more variations and architectures available. The choice of method depends on the specific problem and data at hand.

## Deep Learning by R 
some notes: 
### Change your environment to the python environment

The `reticulate::use_condaenv()` function in R is used to specify and activate a specific conda environment for Python within R. This function allows you to use Python packages and functionalities from within your R environment.

To use `reticulate::use_condaenv()` and activate a specific conda environment named "python.exe", you can use the following code:

```Ruby
library(reticulate)
use_condaenv("python.exe")
```

Make sure that you have the `reticulate` package installed in your R environment before using this function. Also, ensure that the conda environment named "python.exe" exists and is properly configured with the desired Python version and packages.

After running `use_condaenv()`, you should be able to use Python functionality and packages within your R code by using `reticulate::` prefix to access Python objects, modules, and functions. For example, you can use `reticulate::import()` to import Python modules and `reticulate::py_function()` to call Python functions.

Note: Replace "python.exe" with the actual name of your conda environment if it's different.


### Note: The python libraries could be installed using CMD or Terminal window in RStudio. 

- For runnign Deep Learning codes: run the following codes in CMD OR R terminal window in RStudio,
```Ruby
pip install keras
pip install tensorflow
```

-Install keras in R

```Ruby
install.packages("keras")
library(keras)
```

### This article is aimed at applying followinf deep learning models using R:
- Simple Artificial Neural Network (ANN)
- - [Code](https://github.com/hasanmisaii/Deep-Learning-Using-R/blob/10896df56f8b1e9f4e6170d3bd488f4a49e4ebfa/Simple_ANN.R) 
- - [Detail](https://github.com/hasanmisaii/Deep-Learning-Using-R/wiki/Simple-Artificial-Neural-Network-Layer)
- Recurrent Neural Network (RNN)
- - [Code](https://github.com/hasanmisaii/Deep-Learning-Using-R/blob/32ccaf898690bdf112e11a7257632db992f85ee2/RNN.R) 
- - [Detail](https://github.com/hasanmisaii/Deep-Learning-Using-R/wiki/Simple-Recurrent-Neural-Network-Layer)
- Bidirectional Recurrent Neural Network (Bi-RNN)
- - [Code](https://github.com/hasanmisaii/Deep-Learning-Using-R/blob/da3982404789219e458f8231e3ed52af591801cc/Bi-RNN.R) 
- - [Detail](https://github.com/hasanmisaii/Deep-Learning-Using-R/wiki/Bidirectional-Recurrent-Neural-Network-(Bi-RNN))
- Long Short Term Memory (LSTM)
- - [Code](https://github.com/hasanmisaii/Deep-Learning-Using-R/blob/ee2c6555e914f243c4e3c02842b168b70e89952c/LSTM.R) 
- - [Detail](https://github.com/hasanmisaii/Deep-Learning-Using-R/wiki/Long-Short-Term-Memory-Layer)
- Bidirectional Long Short Term Memory (BiLSTM)
- - [Code](https://github.com/hasanmisaii/Deep-Learning-Using-R/blob/b76e505b5e4951ec3cbf50fbe45acd3af7251123/BiLSTM.R)
- - [Detail](https://github.com/hasanmisaii/Deep-Learning-Using-R/wiki/Bidirectional-Long-Short-Term-Memory-(BiLSTM))
