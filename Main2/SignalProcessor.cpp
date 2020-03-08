#include "SignalProcessor.h"

SignalBuffer_t allocate_signal_buffer(size_t size)
{
	SignalBuffer_t buffer;
	buffer.samples = new cuComplex[size];
	buffer.maxsize = size;
	return buffer;
}

void signal_buffer_from_floats(SignalBuffer_t* buffer, float* real, float* imaginary, size_t channels, size_t size)
{
	buffer->size = size;
	buffer->channels = channels;
	for (size_t i = 0; i < size; i++)
	{
		buffer->samples[i] = make_cuComplex(real[i], imaginary[i]);
	}
}

void signal_buffer_to_floats(SignalBuffer_t buffer, float* real, float* imaginary, size_t channel)
{
	size_t channel_size = get_channel_buffer_size(buffer);
	for (size_t i = 0; i < channel_size; i++)
	{
		cuComplex sample = get_signal_buffer_sample(buffer, channel, i);
		real[i] = sample.x;
		imaginary[i] = sample.y;
	}
}

void deallocate_signal_buffer(SignalBuffer_t* buffer)
{
	delete[] buffer->samples;
}

void reallocate_signal_buffer(SignalBuffer_t* buffer, size_t maxsize)
{
	cuComplex* oldS = buffer->samples;
	cuComplex* newS = new cuComplex[maxsize];
	size_t oldSize = buffer->maxsize;

	for (size_t i = 0; i < maxsize && i < oldSize; i++)
	{
		newS[i] = oldS[i];
	}

	buffer->samples = newS;
	buffer->maxsize = maxsize;

	delete[] oldS;
}




SignalProcessor::SignalProcessor(size_t datalen)
{
	this->datalen = datalen;
}

SignalProcessor::~SignalProcessor(){}

size_t SignalProcessor::getDataLen()
{
	return this->datalen;
}

void SignalProcessor::reset(){}
void SignalProcessor::process_buffer(SignalBuffer_t* buffer, size_t channel){}

int SignalProcessor::has_extra_samples()
{
	return 0;
}

void SignalProcessor::get_extra_samples(SignalBuffer_t* buffer, size_t channel)
{
}




