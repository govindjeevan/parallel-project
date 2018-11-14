//standard includes
#include <iostream>
#include <vector>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <omp.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand.h>
#include <curand_kernel.h>

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "CycleTimer.h"

//include definition file
#include "neuralNetwork.h"

using namespace std;

__global__ void
forward_prop_w1(double *device_output, double *input, double *weights, int num_inputs, int num_hidden) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// accessing in transposed manner
	int row = index%num_inputs;
	int col = index/num_inputs;

    
    if (index < num_inputs*num_hidden) {
    	// printf("INPUT %f\n", input[row]);
    	// printf("weights %f\n", weights[row*num_hidden + col]);
    	device_output[index] = input[row]*weights[row*num_hidden + col];
    	// printf("DEVICE %f\n", device_output[index]);
    }
}

__global__ void
forward_prop_w2(double *device_output, double *hidden, double *weights, int num_hidden, int num_outputs) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	// accessing in transposed manner
	int row = index%num_hidden;
	int col = index/num_hidden;

    
    if (index < num_hidden*num_outputs) {
    	device_output[index] = hidden[row]*weights[row*num_outputs + col];
    }
}

__global__ void
fill_in_hidden(double *device_output, double *seg_scanned, int num_input, int num_hidden) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < num_hidden) {
    	device_output[index] = seg_scanned[index*(num_input+1)+num_input];
    }
}


__global__ void
weights1_kernel(double *device_output, int num_inputs, int num_hidden) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	curandState state;
	unsigned int seed = index;
	curand_init(seed, 0, 0, &state);
	float rand = curand_uniform(&state);
    
    if (index < num_inputs*num_hidden) {
    	device_output[index] = ( (double)(rand)/10 - 0.05);
    }
}

__global__ void
weights2_kernel(double *device_output, int num_hidden, int num_outputs) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	curandState state;
	unsigned int seed = index;
	curand_init(seed, 0, 0, &state);
	float rand = curand_uniform(&state);

    if (index < num_outputs*num_hidden) {
    	device_output[index]  = ( (double)(rand)/10 - 0.05);
    }
}

/*******************************************************************
* Constructor
********************************************************************/
neuralNetwork::neuralNetwork(int nI, int nH, int nO) : nInput(nI), nHidden(nH), nOutput(nO)
{				
	//create neuron lists
	//--------------------------------------------------------------------------------------------------------
	inputNeurons = new( float[nInput + 1] );
	for ( int i=0; i < nInput; i++ ) inputNeurons[i] = 0;

	//create input bias neuron
	inputNeurons[nInput] = -1;

	hiddenNeurons = new( float[nHidden + 1] );
	for ( int i=0; i < nHidden; i++ ) hiddenNeurons[i] = 0;

	//create hidden bias neuron
	hiddenNeurons[nHidden] = -1;

	outputNeurons = new( float[nOutput] );
	for ( int i=0; i < nOutput; i++ ) outputNeurons[i] = 0;

	//create weight lists (include bias neuron weights)
	//--------------------------------------------------------------------------------------------------------
	wInputHidden = new( float*[nInput + 1] );
	wInputHidden[0] = new (float[(nInput + 1)*nHidden]);
	for ( int i=1; i <= nInput; i++ ) {
		wInputHidden[i] = wInputHidden[i-1] + nHidden;
	}
	for ( int i=0; i <= nInput; i++ ) 
	{
		for ( int j=0; j < nHidden; j++ ) wInputHidden[i][j] = 0;		
	}

	wHiddenOutput = new( float*[nHidden + 1] );
	wHiddenOutput[0] = new (float[(nHidden + 1)*nOutput]);
	for ( int i=1; i <= nHidden; i++ ) {
		wHiddenOutput[i] = wHiddenOutput[i-1] + nOutput;
	}
	for ( int i=0; i <= nHidden; i++ ) 
	{
		for ( int j=0; j < nOutput; j++ ) wHiddenOutput[i][j] = 0;		
	}	
	
	//initialize weights
	//--------------------------------------------------------------------------------------------------------
	initializeWeights();			
}

/*******************************************************************
* Destructor
********************************************************************/
neuralNetwork::~neuralNetwork()
{
	//delete neurons
	delete[] inputNeurons;
	delete[] hiddenNeurons;
	delete[] outputNeurons;

	//delete weight storage
	for (int i=0; i <= nInput; i++) delete[] wInputHidden[i];
	delete[] wInputHidden;

	for (int j=0; j <= nHidden; j++) delete[] wHiddenOutput[j];
	delete[] wHiddenOutput;	
}

/*******************************************************************
* Save Neuron Weights
********************************************************************/
bool neuralNetwork::saveWeights(char* filename)
{
	//open file for reading
	fstream outputFile;
	outputFile.open(filename, ios::out);

	if ( outputFile.is_open() )
	{
		outputFile.precision(50);		

		//output weights
		for ( int i=0; i <= nInput; i++ ) 
		{
			for ( int j=0; j < nHidden; j++ ) 
			{
				outputFile << wInputHidden[i][j] << ",";				
			}
		}
		
		for ( int i=0; i <= nHidden; i++ ) 
		{		
			for ( int j=0; j < nOutput; j++ ) 
			{
				outputFile << wHiddenOutput[i][j];					
				if ( i * nOutput + j + 1 != (nHidden + 1) * nOutput ) outputFile << ",";
			}
		}

		//print success
		cout << endl << "Neuron weights saved to '" << filename << "'" << endl;

		//close file
		outputFile.close();
		
		return true;
	}
	else 
	{
		cout << endl << "Error - Weight output file '" << filename << "' could not be created: " << endl;
		return false;
	}
}

/*******************************************************************
* Return the NN accuracy on the set
********************************************************************/
double neuralNetwork::getSetAccuracy( std::vector<dataEntry*>& set )
{
	double incorrectResults = 0;
		
	//for every training input array
	for ( int tp = 0; tp < (int) set.size(); tp++)
	{						
		//feed inputs through network and backpropagate errors
		feedForward( set[tp]->pattern );

		int predicted = distance(outputNeurons, max_element(outputNeurons, outputNeurons + nOutput));
		int expected = distance(set[tp]->target, max_element(set[tp]->target, set[tp]->target + nOutput));
		
		if (predicted != expected) incorrectResults++;	
		
	}//end for
	
	//calculate error and return as percentage
	return 100 - (incorrectResults/set.size() * 100);
}

/*******************************************************************
* Initialize Neuron Weights
********************************************************************/
void neuralNetwork::initializeWeights()
{
	for(int i = 0; i <= nInput; i++)
	{		
		for(int j = 0; j < nHidden; j++) 
		{
			//set weights to random values
			wInputHidden[i][j] = ( (( (double)(rand()%1000)+1)/1000)/10 - 0.05);
		}
	}
	
	//set weights between input and hidden
	//--------------------------------------------------------------------------------------------------------
	for(int i = 0; i <= nHidden; i++)
	{		
		for(int j = 0; j < nOutput; j++) 
		{
			//set weights to random values
			wHiddenOutput[i][j] = ( (( (double)(rand()%1000)+1)/1000)/10 - 0.05);
		}
	}

	/*
	double startTime = CycleTimer::currentSeconds();

	int threadsPerBlock = 256;
    int blocks1 = ((nInput+1)*nHidden + threadsPerBlock - 1) / threadsPerBlock;
    int blocks2 = (nOutput*(nHidden+1) + threadsPerBlock - 1) / threadsPerBlock;

    //set weights between input and hidden 		
	//--------------------------------------------------------------------------------------------------------
	double *device_output1;
    cudaMalloc(&device_output1, sizeof(double) * (nInput+1)*nHidden);

	weights1_kernel<<<blocks1, threadsPerBlock>>>(device_output1, nInput+1, nHidden);

	cudaThreadSynchronize();

	cudaMemcpy(wInputHidden[0], device_output1, (nInput+1)*nHidden*sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(device_output1);

	//set weights between input and hidden
	//--------------------------------------------------------------------------------------------------------
	double *device_output2;
	cudaMalloc(&device_output2, sizeof(double) * (nHidden+1)*nOutput);

	weights2_kernel<<<blocks2, threadsPerBlock>>>(device_output2, nHidden+1, nOutput);

	cudaThreadSynchronize();
	
	cudaMemcpy(wHiddenOutput[0], device_output2, (nHidden+1)*nOutput*sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(device_output2);

	double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    printf("Initialize: %f\n", overallDuration);
    */
}
/*******************************************************************
* Activation Function
********************************************************************/
inline float neuralNetwork::activationFunction( float x )
{
	//sigmoid function
	return 1/(1+exp(-x));
}	

/*******************************************************************
* Feed Forward Operation
********************************************************************/
void neuralNetwork::feedForward(float* pattern)
{
	//set input neurons to input values
	for(int i = 0; i < nInput; i++) {
		inputNeurons[i] = pattern[i];
	}
	/*
	int threadsPerBlock = 256;
    int blocks1 = ((nInput+1)*nHidden + threadsPerBlock - 1) / threadsPerBlock;
    
	double *device_output1;
    cudaMalloc(&device_output1, sizeof(double) * (nInput+1)*nHidden);
    double *input;
    cudaMalloc(&input, sizeof(double) * (nInput+1));
    cudaMemcpy(input, inputNeurons, sizeof(double) * (nInput+1), cudaMemcpyHostToDevice);
    double *w1;
    cudaMalloc(&w1, sizeof(double) * (nInput+1)*nHidden);
    cudaMemcpy(w1, wInputHidden[0], (nInput+1)*nHidden*sizeof(double), cudaMemcpyHostToDevice);

	forward_prop_w1<<<blocks1, threadsPerBlock>>>(device_output1, input, w1, nInput+1, nHidden);

	cudaThreadSynchronize();

	double bigArray[(nInput+1)*nHidden];
	cudaMemcpy(bigArray, device_output1, (nInput+1)*nHidden*sizeof(double), cudaMemcpyDeviceToHost);
	// cout << "HI 1 " << bigArray[20] << endl;
	// cout << "HI 2 " << bigArray[21] << endl;
	// cout << "HI 3 " << bigArray[22] << endl;
	// cout << "HI 4 " << bigArray[23] << endl;
	// cout << "HI 5 " << bigArray[24] << endl;

	int keys[(nInput+1)*nHidden];
	for (int t = 0; t<(nInput+1)*nHidden; t++) {
		keys[t] = t / (nInput+1);
	}

	thrust::inclusive_scan_by_key(thrust::host, keys, keys + ((nInput+1)*nHidden), bigArray, bigArray);

	for (int h = 0; h < nHidden; h++) {
		// cout << "big " << bigArray[h*(nInput+1)+nInput] << endl;
		hiddenNeurons[h] = activationFunction(bigArray[h*(nInput+1)+nInput]);
		// cout << "hidden " << hiddenNeurons[h] << endl;
	}
	
	
	cudaFree(device_output1);
	cudaFree(input);
	cudaFree(w1);
	*/

	//inarray is the input scan array, end is the last elem address, result is empty array
	// cudaMemcpy(wInputHidden[0], device_output1, (nInput+1)*nHidden*sizeof(double), cudaMemcpyDeviceToHost);

	 
	//Calculate Hidden Layer values - include bias neuron
	//--------------------------------------------------------------------------------------------------------
	#pragma omp parallel for
	for(int j=0; j < nHidden; j++)
	{
		//clear value
		hiddenNeurons[j] = 0;				
		
		//get weighted sum of pattern and bias neuron

		for( int i=0; i <= nInput; i++ ) {
			hiddenNeurons[j] += inputNeurons[i] * wInputHidden[i][j];
		}
		// cout << "hidden big " << hiddenNeurons[j] << endl;
		
		//set to result of sigmoid
		hiddenNeurons[j] = activationFunction( hiddenNeurons[j] );		

		// cout << "activate " << hiddenNeurons[j] << endl;	
	}
	
	/*
	double *device_output2;
    cudaMalloc(&device_output2, sizeof(double) * (nHidden+1)*nOutput);
    double *hidden;
    cudaMalloc(&hidden, sizeof(double) * (nHidden+1));
    cudaMemcpy(hidden, hiddenNeurons, sizeof(double) * (nHidden+1), cudaMemcpyHostToDevice);
    double *w2;
    cudaMalloc(&w2, sizeof(double) * (nHidden+1)*nOutput);
    cudaMemcpy(w2, wHiddenOutput[0], (nHidden+1)*nOutput*sizeof(double), cudaMemcpyHostToDevice);

	forward_prop_w2<<<blocks1, threadsPerBlock>>>(device_output2, hidden, w2, nHidden+1, nOutput);

	cudaThreadSynchronize();

	double bigArray2[(nHidden+1)*nOutput];
	cudaMemcpy(bigArray2, device_output2, (nHidden+1)*nOutput*sizeof(double), cudaMemcpyDeviceToHost);
	// cout << "HI 1 " << bigArray[20] << endl;
	// cout << "HI 2 " << bigArray[21] << endl;
	// cout << "HI 3 " << bigArray[22] << endl;
	// cout << "HI 4 " << bigArray[23] << endl;
	// cout << "HI 5 " << bigArray[24] << endl;

	int keys2[(nHidden+1)*nOutput];
	for (int q = 0; q<(nHidden+1)*nOutput; q++) {
		keys2[q] = q / (nHidden+1);
	}

	thrust::inclusive_scan_by_key(thrust::host, keys2, keys2 + ((nHidden+1)*nOutput), bigArray2, bigArray2);

	for (int o = 0; o < nOutput; o++) {
		outputNeurons[o] = activationFunction(bigArray2[o*(nHidden+1)+nHidden]);
	}

	cudaFree(device_output2);
	cudaFree(hidden);
	cudaFree(w2);

	*/

	//Calculating Output Layer values - include bias neuron
	//--------------------------------------------------------------------------------------------------------
	// #pragma omp parallel for
	for(int k=0; k < nOutput; k++)
	{
		//clear value
		outputNeurons[k] = 0;				
		
		//get weighted sum of pattern and bias neuron
		for( int j=0; j <= nHidden; j++ ) outputNeurons[k] += hiddenNeurons[j] * wHiddenOutput[j][k];
		
		//set to result of sigmoid
		outputNeurons[k] = activationFunction( outputNeurons[k] );
	}
}

void neuralNetwork::printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}

