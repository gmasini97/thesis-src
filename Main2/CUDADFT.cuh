#pragma once

#include "CUDASignalProcessor.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "CmplxUtils.cuh"

class CUDADFT : public CUDASignalProcessor
{
private:
	SignalBuffer_t tmp;
public:
	CUDADFT(size_t datalen);
	~CUDADFT();
protected:
	void exec_kernel(SignalBuffer_t* host_buffer, SignalBuffer_t* device_buffer, size_t channel);
};