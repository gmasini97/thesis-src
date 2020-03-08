#include "CUDASignalProcessor.cuh"

cudaError_t transfer_buffer_host_to_device(SignalBuffer_t* device, SignalBuffer_t host) {
	cudaError_t status;
	
	device->channels = host.channels;
	device->size = host.size;

	status = cudaMemcpy(device->samples, host.samples, device->size * sizeof(device->samples[0]), cudaMemcpyHostToDevice);
	
	return status;
}

cudaError_t transfer_buffer_device_to_host(SignalBuffer_t* host, SignalBuffer_t device) {
	cudaError_t status;

	//host->channels = device->channels;
	host->channels = device.channels;

	//host->size = device->size;
	host->size = device.size;

	status = cudaMemcpy(host->samples, device.samples, host->size * sizeof(host->samples[0]), cudaMemcpyDeviceToHost);

	return status;
}

cudaError_t cuda_allocate_signal_buffer(SignalBuffer_t* buffer, size_t datalen)
{
	cudaError_t status;

	cuComplex* samples;
	status = cudaMalloc((void**) & (samples), datalen * sizeof(cuComplex));
	if (status != cudaSuccess) return status;

	buffer->samples = samples;
	buffer->maxsize = datalen;

	return status;
}

void cuda_deallocate_signal_buffer(SignalBuffer_t* buffer)
{
	cudaFree(buffer->samples);
}

CUDASignalProcessor::CUDASignalProcessor(size_t datalen) : SignalProcessor(datalen)
{
	cudaError_t status;

	this->err = 0;
	this->time = 0;

	status = cudaSetDevice(0);
	if (check_cuda_status(status, "set_device")) goto fin;

	status = cuda_allocate_signal_buffer(&(this->device_buffer), datalen);
	if (check_cuda_status(status, "alloc_dev_buffer")) goto fin;

fin:
}

CUDASignalProcessor::~CUDASignalProcessor()
{
	cuda_deallocate_signal_buffer(&(this->device_buffer));
}

int CUDASignalProcessor::check_cuda_status(cudaError_t status, const char* msg)
{
	if (status != cudaSuccess)
	{
		this->err = 1;
		std::cout << "[" << msg << "] cuda err: " << cudaGetErrorString(status) << std::endl;
	}
	return this->err;
}

void CUDASignalProcessor::process_buffer(SignalBuffer_t* buffer, size_t channel)
{
	cudaError_t status;
	size_t size = get_channel_buffer_size(*buffer);

	auto start = std::chrono::high_resolution_clock::now();

	status = transfer_buffer_host_to_device(&(this->device_buffer), *buffer);
	if (check_cuda_status(status, "buffer h>d")) return;

	this->exec_kernel(buffer, &(this->device_buffer), channel);

	status = cudaGetLastError();
	if (check_cuda_status(status, "last_err")) return;

	status = cudaDeviceSynchronize();
	if (check_cuda_status(status, "dev_synch")) return;

	status = transfer_buffer_device_to_host(buffer, this->device_buffer);
	if (check_cuda_status(status, "buffer d>h")) return;

	auto end = std::chrono::high_resolution_clock::now();
	
	this->time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

}

void CUDASignalProcessor::reset()
{
	this->time = 0;
}

int CUDASignalProcessor::get_err()
{
	return this->err;
}

long long int CUDASignalProcessor::get_time()
{
	return this->time;
}