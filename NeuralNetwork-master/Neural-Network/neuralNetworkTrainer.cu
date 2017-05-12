//standard includes
#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <omp.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <cublas_v2.h>

//include definition file
#include "neuralNetworkTrainer.h"

#include "CycleTimer.h"

using namespace std;


// void gpu_blas_mmul(cublasHandle_t &handle, const float *A, const float *B, float *C, const int m, const int k, const int n) {
// 	int lda=m, ldb=k, ldc=m;
// 	const float alf =1;
// 	const float bet =0;
// 	const float *alpha = &alf; 
// 	const float *beta =&bet;


// 	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
// }

__global__ void
back_prop_kernel(float *device_output, float *input, float *hidden, float* w2, float* outputErrorGradients, int nInput, int nHidden, int nOutput,  float learningRate) {
	int linearThreadIndex = threadIdx.x;
	int unit = blockIdx.x;
    // should we compute this outside of here, or make this parallel somehow as well? dedciate #output threads to dothis?
    __shared__ float weightedSum[1];
    if (linearThreadIndex==0) {
        for (int i=0; i<nOutput; i++) {
          weightedSum[0] += w2[unit*nOutput + i] * outputErrorGradients[i];
        }
    }

    __syncthreads();

    if (linearThreadIndex < nInput) {
    	// printf("LR: %f\n",learningRate );
    	// printf("INPUT: %f\n",input[linearThreadIndex] );
    	// printf("HIDDEN: %f\n",hidden[unit] );
    	// printf("WS: %f\n",weightedSum[0] );

        device_output[linearThreadIndex*nHidden + unit] = learningRate * input[linearThreadIndex] * hidden[unit]*(1 - hidden[unit]) * weightedSum[0];
    }
}

__global__ void 
back_prop_kernel_batch(float *device_output, float *input, float *hidden, float* w2, float* outputErrorGradients, int nInput, int nHidden, int nOutput, float learningRate, int batchSize) {
    int linearThreadIndex = threadIdx.x;
    int unit = blockIdx.x%nHidden;
    int batch = blockIdx.x/nHidden; 
    
    __shared__ float weightedSum[1];
    float temp = 0.0;
    if (linearThreadIndex ==0 && unit<nHidden) {
        for (int i=0; i<nOutput; i++) { 
            weightedSum[0] += w2[unit*nOutput + i] * outputErrorGradients[batch*(nOutput+1) +i];
        }
    }
    
    __syncthreads();
   
    if (linearThreadIndex < nInput) {
        temp = learningRate * input[batch*(nInput+1) + linearThreadIndex] * hidden[batch*(nHidden+1) + unit]*(1 - hidden[batch*(nHidden+1) + unit]) * weightedSum[0];
        atomicAdd(&device_output[linearThreadIndex*nHidden + unit], temp);
    } 

}

/*******************************************************************
* constructor
********************************************************************/
neuralNetworkTrainer::neuralNetworkTrainer( neuralNetwork *nn )	:	NN(nn),
																	epoch(0),
																	learningRate(LEARNING_RATE),
																	maxEpochs(MAX_EPOCHS),
																	desiredAccuracy(DESIRED_ACCURACY),																	
																	useBatch(false),
																	trainingSetAccuracy(0),
																	validationSetAccuracy(0),
																	generalizationSetAccuracy(0)																	
{
	//create delta lists
	//--------------------------------------------------------------------------------------------------------


	deltaInputHidden = new( float*[NN->nInput + 1] );
    deltaInputHidden[0] = new (float[((NN->nInput) + 1)*(NN->nHidden)]);
    for ( int i=1; i <= NN->nInput; i++ ) {
		deltaInputHidden[i] = deltaInputHidden[i-1] + NN->nHidden;
	}

	for ( int i=0; i <= NN->nInput; i++ ) 
	{
		for ( int j=0; j < NN->nHidden; j++ ) deltaInputHidden[i][j] = 0;		
	}

	/*for ( int i=0; i <= NN->nInput; i++ ) 
	{
		deltaInputHidden[i] = new (float[NN->nHidden]);
		for ( int j=0; j < NN->nHidden; j++ ) deltaInputHidden[i][j] = 0;		
	}*/

	deltaHiddenOutput = new( float*[NN->nHidden + 1] );
	for ( int i=0; i <= NN->nHidden; i++ ) 
	{
		deltaHiddenOutput[i] = new (float[NN->nOutput]);			
		for ( int j=0; j < NN->nOutput; j++ ) deltaHiddenOutput[i][j] = 0;		
	}

	//create error gradient storage
	//--------------------------------------------------------------------------------------------------------
	hiddenErrorGradients = new( float[(NN->batchSize)*(NN->nHidden + 1)] );
	for (int b = 0; b<NN->batchSize; b++) {
	    for(int i = 0; i < NN->nHidden+1; i++) { 
            hiddenErrorGradients[b*(NN->nHidden+1) + i] = 0;
        }
	}
	
	outputErrorGradients = new( float[(NN->batchSize)*(NN->nOutput + 1)] );
	for (int b = 0; b<NN->batchSize; b++) {
	    for(int i = 0; i < NN->nOutput+1; i++) { 
            outputErrorGradients[b*(NN->nOutput+1) + i] = 0;
        }
	}

    cudaMalloc(&device_output1, sizeof(float) * (NN->batchSize)*((NN->nInput)+1)*(NN->nHidden));
    cudaMalloc(&input, sizeof(float) * (NN->batchSize)*((NN->nInput)+1));
    cudaMalloc(&hidden, sizeof(float) * (NN->batchSize)*(((NN->nHidden) +1)));
    cudaMalloc(&w2, sizeof(float) * ((NN->nHidden)+1)*(NN->nOutput));
    cudaMalloc(&output_error_gradients, sizeof(float)*((NN->nOutput) +1));

	// hiddenErrorGradients = new( float[NN->nHidden + 1] );
	// for ( int i=0; i <= NN->nHidden; i++ ) hiddenErrorGradients[i] = 0;
	
	// outputErrorGradients = new( float[NN->nOutput + 1] );
	// for ( int i=0; i <= NN->nOutput; i++ ) outputErrorGradients[i] = 0;
}


/*******************************************************************
* Set training parameters
********************************************************************/
void neuralNetworkTrainer::setTrainingParameters( double lR, bool batch )
{
	learningRate = lR;
	useBatch = batch;
}
/*******************************************************************
* Set stopping parameters
********************************************************************/
void neuralNetworkTrainer::setStoppingConditions( int mEpochs, double dAccuracy )
{
	maxEpochs = mEpochs;
	desiredAccuracy = dAccuracy;	
}
/*******************************************************************
* Enable training logging
********************************************************************/
void neuralNetworkTrainer::enableLogging(const char* filename, int resolution = 1)
{
	//create log file 
	if ( ! logFile.is_open() )
	{
		logFile.open(filename, ios::out);

		if ( logFile.is_open() )
		{
			//write log file header
			logFile << "Epoch,Training Set Accuracy, Generalization Set Accuracy,Training Set MSE, Generalization Set MSE" << endl;
			
			//enable logging
			loggingEnabled = true;
			
			//resolution setting;
			logResolution = resolution;
			lastEpochLogged = -resolution;
		}
	}
}
/*******************************************************************
* calculate output error gradient
********************************************************************/
inline float neuralNetworkTrainer::getOutputErrorGradient( float desiredValue, float outputValue)
{
	//return error gradient
	return outputValue * ( 1 - outputValue ) * ( desiredValue - outputValue );
}

/*******************************************************************
* calculate input error gradient
********************************************************************/
float neuralNetworkTrainer::getHiddenErrorGradient( int j )
{
	//get sum of hidden->output weights * output error gradients
	float weightedSum = 0;
	for( int k = 0; k < NN->nOutput; k++ ) {
		weightedSum += NN->wHiddenOutput[j][k] * outputErrorGradients[k];
	}

	return NN->hiddenNeurons[j] * ( 1 - NN->hiddenNeurons[j] ) * weightedSum;
}

float neuralNetworkTrainer::getHiddenErrorGradientBatch( int j, int b )
{
	//get sum of hidden->output weights * output error gradients
	float weightedSum = 0;
	for( int k = 0; k < NN->nOutput; k++ ) {
		weightedSum += NN->wHiddenOutput[j][k] * outputErrorGradients[b*k];
	}

	return NN->hiddenNeurons[j] * ( 1 - NN->hiddenNeurons[j] ) * weightedSum;
}
/*******************************************************************
* Train the NN using gradient descent
********************************************************************/
void neuralNetworkTrainer::trainNetwork( trainingDataSet* tSet )
{
	cout	<< endl << " Neural Network Training Starting: " << endl
			<< "==========================================================================" << endl
			<< " LR: " << learningRate << ", Max Epochs: " << maxEpochs << ", Batch: " << useBatch << endl
			<< " " << NN->nInput << " Input Neurons, " << NN->nHidden << " Hidden Neurons, " << NN->nOutput << " Output Neurons" << endl
			<< "==========================================================================" << endl << endl;

	//reset epoch and log counters
	epoch = 0;
	lastEpochLogged = -logResolution;
		
	//train network using training dataset for training and generalization dataset for testing
	//--------------------------------------------------------------------------------------------------------
	while (	( trainingSetAccuracy < desiredAccuracy || generalizationSetAccuracy < desiredAccuracy ) && epoch < maxEpochs )				
	{			
		//store previous accuracy
		double previousTAccuracy = trainingSetAccuracy;
		double previousGAccuracy = generalizationSetAccuracy;

		//use training set to train network
		runTrainingEpoch( tSet->trainingSet );

		// trainingSetAccuracy = NN->getSetAccuracy( tSet->trainingSet );
		//get generalization set accuracy
		generalizationSetAccuracy = NN->getSetAccuracy( tSet->generalizationSet );

		//Log Training results
		if ( loggingEnabled && logFile.is_open() && ( epoch - lastEpochLogged == logResolution ) ) 
		{
			logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << endl;
			lastEpochLogged = epoch;
		}
		
		//print out change in training /generalization accuracy (only if a change is greater than a percent)
		if ( ceil(previousTAccuracy) != ceil(trainingSetAccuracy) || ceil(previousGAccuracy) != ceil(generalizationSetAccuracy) ) 
		{	
			cout << "Epoch :" << epoch;
			cout << " TSet Acc:" << trainingSetAccuracy << "%" ;
			cout << " GSet Acc:" << generalizationSetAccuracy << "%" << endl;
		}
		
		//once training set is complete increment epoch
		epoch++;

	}//end while

	//get validation set accuracy
	validationSetAccuracy = NN->getSetAccuracy(tSet->validationSet);

	//log end
	logFile << epoch << "," << trainingSetAccuracy << "," << generalizationSetAccuracy << endl << endl;
	logFile << "Training Complete!!! - > Elapsed Epochs: " << epoch << " Validation Set Accuracy: " << validationSetAccuracy << endl;
			
	//out validation accuracy
	cout << endl << "Training Complete!!! - > Elapsed Epochs: " << epoch << endl;
	cout << " Validation Set Accuracy: " << validationSetAccuracy << endl << endl;
}
/*******************************************************************
* Run a single training epoch
********************************************************************/
void neuralNetworkTrainer::runTrainingEpoch( vector<dataEntry*> trainingSet )
{
	double startIter = CycleTimer::currentSeconds();
	//incorrect patterns
	double incorrectPatterns = 0;
	
	vector<float*>largePattern;
	vector<float*>largeTarget; 

	double startForward;
	double endForward;

	double startBack;
	double endBack;


	//for every training pattern
	for ( int tp = 0; tp < (int) trainingSet.size(); tp++)
	{
		largePattern.push_back(trainingSet[tp]->pattern);
		largeTarget.push_back(trainingSet[tp]->target);
		//feed inputs through network and backpropagate errors
		// double startForward = CycleTimer::currentSeconds();
		if (useBatch && ((tp == (int) trainingSet.size()-1) || (largePattern.size() == NN->batchSize))) {
			startForward = CycleTimer::currentSeconds();
			NN->feedForwardBatch( largePattern );
			endForward = CycleTimer::currentSeconds();

			startBack = CycleTimer::currentSeconds();
			backpropagateBatch( largeTarget );
			endBack = CycleTimer::currentSeconds();

			updateWeights();
			largePattern.clear();
		} else {
			startForward = CycleTimer::currentSeconds();
			NN->feedForward( trainingSet[tp]->pattern );
			endForward = CycleTimer::currentSeconds();

			startBack = CycleTimer::currentSeconds();
			backpropagate( trainingSet[tp]->target );
			endBack = CycleTimer::currentSeconds();
		}

		
	    double timeForward = endForward - startForward;

	    // printf("Forward: %f\n", timeForward);

	    double timeBack = endBack - startBack;

	    printf("Backprop: %f\n", timeBack);

	    double timeBoth = endBack - startForward;

	    // printf("Both: %f\n", timeBoth);


		int predicted = distance(NN->outputNeurons, max_element(NN->outputNeurons, NN->outputNeurons + NN->nOutput));
		int expected = distance(trainingSet[tp]->target, max_element(trainingSet[tp]->target, trainingSet[tp]->target + NN->nOutput));
		
		if (predicted != expected) incorrectPatterns++;
			
		
	}//end for

	//if using batch learning - update the weights
	// if ( useBatch ) updateWeights();
	
	//update training accuracy
	trainingSetAccuracy = 100 - (incorrectPatterns/trainingSet.size() * 100);

	double endIter = CycleTimer::currentSeconds();
    double timeIter = endIter - startIter;

    printf("Iteration: %f\n", timeIter);

}

/*******************************************************************
* Propagate errors back through NN and calculate delta values in batches
********************************************************************/
void neuralNetworkTrainer::backpropagateBatch(vector<float*> desiredOutputsVector) {
	for (int b=0; b<NN->batchSize; b++) {
		// #pragma omp parallel
		// {
			// #pragma omp for
			for (int k = 0; k < (NN->nOutput); k++)
			{
				// cout << "TNUM " << omp_get_thread_num() << endl;
				//get error gradient for every output node
				//outputErrorGradients[k] = getOutputErrorGradient( desiredOutputsVector[b][k], NN->outputNeurons[b*k] );
				outputErrorGradients[b*k] = getOutputErrorGradient( desiredOutputsVector[b][k], NN->outputNeurons[b*k] );

				//for all nodes in hidden layer and bias neuron
				// #pragma omp for
				for (int j = 0; j <= NN->nHidden; j++) 
				{
					// if (omp_get_thread_num()) {
					// 	cout << "TNUM " << omp_get_thread_num() << endl;
					// }
					//calculate change in weight
					// #pragma omp atomic
					deltaHiddenOutput[j][k] += learningRate * NN->hiddenNeurons[b*j] * outputErrorGradients[b*k];
				}
			}
			
			// #pragma omp for
			for (int j = 0; j < NN->nHidden; j++)
			{
				//get error gradient for every hidden node
				hiddenErrorGradients[b*j] = getHiddenErrorGradientBatch( j, b );

				//for all nodes in input layer and bias neuron
				// #pragma omp for
				for (int i = 0; i <= NN->nInput; i++)
				{
					//calculate change in weight 
					// #pragma omp atomic
					deltaInputHidden[i][j] += learningRate * NN->inputNeurons[b*i] * hiddenErrorGradients[b*j]; 

				}
			}
			
			
		// }
	}
/*
	dim3 blockDim(1024, 1);
    dim3 gridDim(NN->nHidden);
    cudaMemcpy(input, NN->inputNeurons, sizeof(float) * ((NN->nInput)+1) *(NN->batchSize), cudaMemcpyHostToDevice);
    cudaMemcpy(hidden, NN->hiddenNeurons, (NN->batchSize)*((NN->nHidden)+1)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(w2, NN->wHiddenOutput[0], ((NN->nHidden)+1)*(NN->nOutput)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_error_gradients, outputErrorGradients, sizeof(float) * (NN->batchSize)*((NN->nOutput)+1), cudaMemcpyHostToDevice);
    back_prop_kernel<<<gridDim, blockDim>>>(device_output1, input, hidden, w2, output_error_gradients, (NN->nInput)+1, NN->nHidden, NN->nOutput, learningRate);
    cudaMemcpy(deltaInputHidden[0], device_output1, ((NN->nInput) +1)*(NN->nHidden)*sizeof(float), cudaMemcpyDeviceToHost);
*/
}

/*******************************************************************
* Propagate errors back through NN and calculate delta values
********************************************************************/
void neuralNetworkTrainer::backpropagate( float* desiredOutputs )
{	
	// #pragma omp parallel
	// {
		//modify deltas between hidden and output layers
		//--------------------------------------------------------------------------------------------------------
		// #pragma omp for
		for (int k = 0; k < NN->nOutput; k++)
		{
			// cout << "TNUM " << omp_get_thread_num() << endl;
			//get error gradient for every output node
			outputErrorGradients[k] = getOutputErrorGradient( desiredOutputs[k], NN->outputNeurons[k] );
			
			//for all nodes in hidden layer and bias neuron
			// #pragma omp for
			for (int j = 0; j <= NN->nHidden; j++) 
			{
				// if (omp_get_thread_num()) {
				// 	cout << "TNUM " << omp_get_thread_num() << endl;
				// }
				//calculate change in weight
				if ( !useBatch ) deltaHiddenOutput[j][k] = learningRate * NN->hiddenNeurons[j] * outputErrorGradients[k];
				else deltaHiddenOutput[j][k] += learningRate * NN->hiddenNeurons[j] * outputErrorGradients[k];
			}
		}
		//modify deltas between input and hidden layers
		//--------------------------------------------------------------------------------------------------------
		/*
		// #pragma omp for
		for (int j = 0; j < NN->nHidden; j++)
		{
			//get error gradient for every hidden node
			hiddenErrorGradients[j] = getHiddenErrorGradient( j );

			//for all nodes in input layer and bias neuron
			// #pragma omp for
			for (int i = 0; i <= NN->nInput; i++)
			{
				//calculate change in weight 
				if ( !useBatch ) deltaInputHidden[i][j] = learningRate * NN->inputNeurons[i] * hiddenErrorGradients[j];
				else deltaInputHidden[i][j] += learningRate * NN->inputNeurons[i] * hiddenErrorGradients[j]; 

			}
		}
		*/
		
		
	// }

	
	
	dim3 blockDim(1024, 1);
    dim3 gridDim(NN->nHidden);
    cudaMemcpy(input, NN->inputNeurons, sizeof(float) * ((NN->nInput)+1), cudaMemcpyHostToDevice);
    cudaMemcpy(hidden, NN->hiddenNeurons, ((NN->nHidden)+1)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(w2, NN->wHiddenOutput[0], ((NN->nHidden)+1)*(NN->nOutput)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(output_error_gradients, outputErrorGradients, sizeof(float) * ((NN->nOutput)+1), cudaMemcpyHostToDevice);
    back_prop_kernel<<<gridDim, blockDim>>>(device_output1, input, hidden, w2, output_error_gradients, (NN->nInput)+1, NN->nHidden, NN->nOutput, learningRate);
    cudaMemcpy(deltaInputHidden[0], device_output1, (NN->batchSize)*(NN->nInput +1)*(NN->nHidden)*sizeof(float), cudaMemcpyDeviceToHost);
	

	/*
	cublasHandle_t handle;
	cublasCreate(&handle);

	gpu_blas_mmul(handle, input, w1, device_output1, 1, nInput+1, nHidden);

	cudaMemcpy(hiddenNeurons, device_output1, nHidden*sizeof(float), cudaMemcpyDeviceToHost);

	cublasDestroy(handle);
	*/
	//if using stochastic learning update the weights immediately
	if ( !useBatch ) updateWeights();
}
/*******************************************************************
* Update weights using delta values
********************************************************************/
void neuralNetworkTrainer::updateWeights()
{
	//input -> hidden weights
	//--------------------------------------------------------------------------------------------------------
	for (int i = 0; i <= NN->nInput; i++)
	{
		for (int j = 0; j < NN->nHidden; j++) 
		{
			//update weight
			NN->wInputHidden[i][j] += deltaInputHidden[i][j];	
			
			//clear delta only if using batch (previous delta is needed for momentum
			if (useBatch) deltaInputHidden[i][j] = 0;				
		}
	}
	
	//hidden -> output weights
	//--------------------------------------------------------------------------------------------------------
	for (int j = 0; j <= NN->nHidden; j++)
	{
		for (int k = 0; k < NN->nOutput; k++) 
		{					
			//update weight
			NN->wHiddenOutput[j][k] += deltaHiddenOutput[j][k];
			
			//clear delta only if using batch (previous delta is needed for momentum)
			if (useBatch)deltaHiddenOutput[j][k] = 0;
		}
	}
}
