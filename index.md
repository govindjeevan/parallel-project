# parallel project


**Team members**: Abhy Vytheeswaran, Gaurav Lahiry 


**Summary**: 
We are going to implement a deep neural network using backpropagation to classify images. We want to use some combination of OpenMP and CUDA, as well as exploit the GPU, to speed up the training and testing phases of the network. We will compare our final implementation to a purely sequential algorithm and other existing models like Caffe.

**Background**
A neural network is a machine learning structure containing layers of nodes that are connected to each other. This includes the input layer, the hidden layers, and the output layer. Our neural network will use a backpropagation algorithm to adjust weights and output a label that predicts the class that a test image belongs to. Neurons in each layer are independent from each other so weights can be computed in parallel.

We will start with a single hidden layer and have a complete network where each layer is completely connected to the previous. We will use the sigmoid function as our activation function and stochastic gradient descent to compute the best way to adjust our weights. This combination will also allow us to not get stuck in local minimums when trying to minimize our error while training. We will train our network on the MNIST training set of 60,000 images of handwritten digits. We will then test it on 10,000 test images.  

There is a need for neural networks to be computationally efficient so that we are able to train and use very powerful classifiers without having to sacrifice a great deal of time to waiting for them to be developed. Thus, we want to find fundamental parts of the algorithm in both forward and back propagation steps where there are bottlenecks due to naive sequential code, and attempt to parallelize these aspects. In addition, it would be nice if we could use a trained model to classify large sets of unlabelled data, through further optimizations such as batch processing and filling up the GPU execution contexts available to us.

**Challenges** 

The challenge exists in the fact that a neural net is overall an iterative process that feeds weights forward through the network one hidden layer at a time, and then reverses the process to adjust the weights depending on which nodes in each layer contributed most to the error in our output. Thus we must find ways of parallelism for the work within each layer. This would mean taking advantage of whatever computations are being done naively that have the opportunity for speedup, such as matrix multiplications and mapping independent operations like applying activation functions to be done in parallel. 

However, each layer must be done sequentially, so we must be sure to synchronize threads in between, so there is an overhead of communication across threads within each layer. Furthermore, the weight matrices will be accessed frequently, which could potentially lead to false sharing, if we are not careful to make sure each thread doesn’t bring elements accessed by other threads up into their cache. 

Through this, we hope to learn the ideas behind parallelizing a neural network for efficient training and classification, as it is important that we are not limited by slow speeds as this would outweigh the benefits of having a more powerful deep learning classifier. In this way, we can get a deeper insight into how complex and efficient models like Caffe are implemented.

**Resources** 
Included are papers that we will refer to when implementing our parallel deep neural network. 

Dahl, George, Alan McAvinney, and Tia Newhall. "Parallelizing neural network training for cluster systems." Proceedings of the IASTED International Conference on Parallel and Distributed Computing and Networks. ACTA Press, 2008.
https://www.cs.swarthmore.edu/~newhall/papers/pdcn08.pdf

Krizhevsky, Alex. “One weird trick for parallelizing convolutional neural networks.”. eprint arXiv:1404.599. 2014.
https://arxiv.org/pdf/1404.5997.pdf 

Podlozhnyuk, Victor. "Image Convolution with CUDA."  
http://developer.download.nvidia.com/assets/cuda/files/convolutionSeparable.pdf.

Pethick, Mark, Michael Liddle, Paul Werstein, and Zhiyi Huang. “Parallelization of a Backpropagation Neural Network on a Cluster Computer.” 2003. 
http://www.cs.otago.ac.nz/staffpriv/hzy/papers/pdcs03.pdf 

**Goals and Deliverables** 
Plan to achieve:
* Achieve linear speedup in training a neural network proportional to number of cores
* Accurate predictions when classifying test images (mean squared error < 20%) using at least 1 hidden layer and a simple sigmoid activation function
* Use OpenMP to parallelize computation across the 8 core Xeon processor and utilize the GTX 1080 GPU using CUDA threads
Present a demo showing our speedups compared to the baseline and comparisons to current state of the art models (e.g. Caffe)

Hope to achieve:
* Achieve greater than linear speedup compared to our sequential implementation
* Highly accurate predictions when classifying test images (error < 10%) by adding more layers
* Further optimizations of classifying large sets of image in parallel using batch processing to achieve higher levels of classification throughput
* Use a shared memory model to improve memory access latency and leverage SIMD execution on the supported AVX2 vectorization
* Implement a distributed version utilizing multiple nodes in a cluster (e.g. latedays)

**Platform choice**
We want to use the GHC cluster machines which contain the NVIDIA GeForce GTX 1080 GPU. This will allow us to implement parallelization on the GPU through CUDA, and potentially use OpenMP to parallelize across the 8 core i7 Xeon processors. Each of these cores also supports AVX2 SIMD execution, which we may also want to utilize to achieve greater speedups, especially in regions of code where we are applying the same instruction to multiple data objects. We think the balance of computing resources provided by these machines will allow us to be able to optimize in different ways and think about various methods of parallelization. 

**Schedule** 
April 17  - Implement sequential neural network to classify images in the MNIST dataset.
April 24 - Parallelize matrix multiplications required to train the neural network with CUDA
May 1 - Use OpenMP to parallelize computation across multiple threads of 8-core Xeon
May 8 - Achieve linear speedup and the desired error rate
May 12 - Add more hidden layers to optimize the speedups and error rates and leverage SIMD execution and batch processing as time allows
