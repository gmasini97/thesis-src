#include "MultichannelSignalProcessor.cuh"


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
			bufferRe[i] = real[dataIndex];
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