#pragma once

#include "SignalProcessor.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>

class CUDASignalProcessor : public SignalProcessor
{
private:
	float* re;
	float* im;
	int err;
	long long int time;
protected:
	int check_cuda_status(cudaError_t status);
	virtual void exec_kernel(float* real, float* imaginary, size_t readcount) = 0;
public:
	CUDASignalProcessor(size_t datalen);
	~CUDASignalProcessor();
	void process_buffer(float* real, float* imaginary, size_t readcount);
	void reset();
	int get_err();
	long long int get_time();
};