#include "pch.h"
#include "SignalProcessor.h"

SignalProcessor::SignalProcessor(size_t datalen)
{
	this->datalen = datalen;
}

size_t SignalProcessor::getDataLen()
{
	return this->datalen;
}








MultichannelSignalProcessor::MultichannelSignalProcessor(size_t datalen, size_t channels, SignalProcessor** processors)
{
	this->processors = processors;
	this->datalen = datalen;
	this->channels = channels;
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

void MultichannelSignalProcessor::process_buffer(float* data, size_t readcount)
{
	size_t channels = this->channels;
	size_t datalen = readcount;
	size_t dataIndex = 0;
	size_t totalSamples = datalen / channels;
	size_t maxSamples = getDataLen() / channels;

	float* buffer = (float*) malloc(sizeof(float)*maxSamples);
	for (size_t channel = 0; channel < channels; channel++)
	{
		for (size_t i = 0; i < totalSamples; i++) {
			dataIndex = channel + i * channels;
			buffer[i] =  data[dataIndex];
		}

		this->processors[channel]->process_buffer(buffer, totalSamples);

		for (size_t i = 0; i < totalSamples; i++) {
			dataIndex = channel + i * channels;
			data[dataIndex] = buffer[i];
		}
	}
}

NoProcessor::NoProcessor(size_t datalen) : SignalProcessor(datalen){};
void NoProcessor::reset(){};
void NoProcessor::process_buffer(float* data, size_t readcount) {};