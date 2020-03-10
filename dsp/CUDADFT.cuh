#pragma once

#include "CUDASignalProcessor.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "CmplxUtils.cuh"

#include "LogUtils.h"

class CUDADFT : public CUDASignalProcessor
{
private:
	SignalBuffer_t tmp;
public:
	CUDADFT(AbstractSignalProcessor* next, BitMask channels_to_process);
	~CUDADFT();
	int init(size_t max_buffer_size, size_t channels);
protected:
	void exec_kernel(SignalBuffer_t* host_buffer, SignalBuffer_t* device_buffer);
};