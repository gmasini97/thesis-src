#pragma once

#include "SignalProcessor.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>
#include "LogUtils.h"


cudaError_t transfer_buffer_host_to_device(SignalBuffer_t* device, SignalBuffer_t host);
cudaError_t transfer_buffer_device_to_host(SignalBuffer_t* host, SignalBuffer_t device);
cudaError_t cuda_allocate_signal_buffer(SignalBuffer_t* buffer, size_t datalen, size_t channels);
cudaError_t cuda_clear_signal_buffer_deep(SignalBuffer_t buffer);
void get_threads_blocks_count(size_t processes, dim3& threadsPerBlock, dim3& blocks);
void cuda_deallocate_signal_buffer(SignalBuffer_t* buffer);


class CUDASignalProcessor : public SignalProcessor
{
private:
	SignalBuffer_t device_buffer;
	int err;
	int parallel_channels;
	long long int time;
protected:
	int check_cuda_status(cudaError_t status, const char* msg = "");
	virtual void exec_kernel(SignalBuffer_t* host_buffer, SignalBuffer_t* device_buffer) = 0;
public:
	CUDASignalProcessor(AbstractSignalProcessor* next, BitMask channels_to_process);
	~CUDASignalProcessor();
	int init(size_t max_buffer_size, size_t channels);
	void process_buffer(SignalBuffer_t* buffer);
	int get_err();
	long long int get_time();
};