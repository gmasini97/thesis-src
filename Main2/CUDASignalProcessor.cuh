#pragma once

#include "SignalProcessor.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>

cudaError_t transfer_buffer_host_to_device(SignalBuffer_t* device, SignalBuffer_t host);
cudaError_t transfer_buffer_device_to_host(SignalBuffer_t* host, SignalBuffer_t device);
cudaError_t cuda_allocate_signal_buffer(SignalBuffer_t* buffer, size_t datalen);
void cuda_deallocate_signal_buffer(SignalBuffer_t* buffer);

class CUDASignalProcessor : public SignalProcessor
{
private:
	SignalBuffer_t device_buffer;
	int err;
	long long int time;
protected:
	int check_cuda_status(cudaError_t status, const char* msg = "");
	virtual void exec_kernel(SignalBuffer_t* host_buffer, SignalBuffer_t* device_buffer, size_t channel) = 0;
public:
	CUDASignalProcessor(size_t datalen);
	~CUDASignalProcessor();
	void process_buffer(SignalBuffer_t* buffer, size_t channel);
	void reset();
	int get_err();
	long long int get_time();
};