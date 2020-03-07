#pragma once

#include "CUDASignalProcessor.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

class CUDADFT : public CUDASignalProcessor
{
private:
	float* reTmp;
	float* imTmp;
public:
	CUDADFT(size_t datalen);
	~CUDADFT();
protected:
	void exec_kernel(float *real, float* imaginary, size_t readcount);
};