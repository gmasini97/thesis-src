#include "pch.h"
#include "SignalProcessor.h"

SignalProcessor::SignalProcessor(size_t datalen)
{
	this->datalen = datalen;
}
SignalProcessor::~SignalProcessor(){}

size_t SignalProcessor::getDataLen()
{
	return this->datalen;
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







MultichannelSignalProcessor::MultichannelSignalProcessor(size_t datalen, size_t channels, SignalProcessor** processors)
{
	this->processors = processors;
	this->datalen = datalen;
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

size_t MultichannelSignalProcessor::getDataLen()
{
	return this->datalen;
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

NoProcessor::NoProcessor(size_t datalen) : SignalProcessor(datalen){};
NoProcessor::~NoProcessor(){};
void NoProcessor::reset(){};
void NoProcessor::process_buffer(float* real, float* imaginary, size_t readcount) {};