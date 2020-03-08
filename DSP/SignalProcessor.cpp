#include "pch.h"
#include "SignalProcessor.h"

SignalProcessor::SignalProcessor(size_t datalen, size_t channel)
{
	this->datalen = datalen;
}
SignalProcessor::~SignalProcessor(){}

size_t SignalProcessor::getDataLen()
{
	return this->datalen;
}

void SignalProcessor::reset(){}
void SignalProcessor::process_buffer(float* real, float* imaginary, size_t channels, size_t readcount)
{
	SignalBuffer_t signalBuffer = {real, imaginary, channels, readcount};
	this->process_buffer(signalBuffer);
}

int SignalProcessor::has_extra_samples()
{
	return 0;
}

size_t SignalProcessor::get_extra_samples(float* real, float* imaginary, size_t buffer_size)
{
	return 0;
}





SignalProcessorChain::SignalProcessorChain(size_t datalen, SignalProcessor** chain, size_t chainlen) : SignalProcessor(datalen)
{

	this->chain = chain;
	this->chainlen = chainlen;
}

SignalProcessorChain::~SignalProcessorChain()
{
	for (size_t x = 0; x < this->chainlen; x++)
	{
		delete this->chain[x];
	}
	delete[] this->chain;
}

void SignalProcessorChain::reset()
{
	for (size_t x = 0; x < this->chainlen; x++)
	{
		this->chain[x]->reset();
	}
}

void SignalProcessorChain::process_buffer(float* real, float* imaginary, size_t readcount)
{
	for (size_t x = 0; x < this->chainlen; x++)
	{
		this->chain[x]->process_buffer(real, imaginary, readcount);
	}
}







MultichannelSignalProcessor::MultichannelSignalProcessor(size_t datalen, size_t channels, SignalProcessor** processors) : SignalProcessor(datalen)
{
	this->processors = processors;
	this->channels = channels;

	this->bufferRe = new float[datalen / channels];
	this->bufferIm = new float[datalen / channels];
}

MultichannelSignalProcessor::~MultichannelSignalProcessor()
{
	for (size_t channel = 0; channel < channels; channel++)
		delete processors[channel];
	free(processors);
	delete[] bufferRe;
	delete[] bufferIm;
}

void MultichannelSignalProcessor::reset()
{
	size_t channels = this->channels;
	for (size_t channel = 0; channel < channels; channel++)
		this->processors[channel]->reset();
}

void MultichannelSignalProcessor::process_buffer(float* real, float* imaginary, size_t readcount)
{
	size_t channels = this->channels;
	size_t datalen = readcount;
	size_t dataIndex = 0;
	size_t totalSamples = datalen / channels;
	size_t maxSamples = getDataLen() / channels;

	for (size_t channel = 0; channel < channels; channel++)
	{
		for (size_t i = 0; i < totalSamples; i++) {
			dataIndex = channel + i * channels;
			bufferRe[i] =  real[dataIndex];
			bufferIm[i] = imaginary[dataIndex];
		}

		this->processors[channel]->process_buffer(bufferRe, bufferIm, totalSamples);

		for (size_t i = 0; i < totalSamples; i++) {
			dataIndex = channel + i * channels;
			real[dataIndex] = bufferRe[i];
			imaginary[dataIndex] = bufferIm[i];
		}
	}
}