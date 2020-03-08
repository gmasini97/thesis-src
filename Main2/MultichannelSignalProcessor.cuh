#pragma once
#include "SignalProcessor.cuh"


class MultichannelSignalProcessor : public SignalProcessor
{
private:
	size_t channels;
	SignalProcessor** processors;
public:
	MultichannelSignalProcessor(size_t datalen, size_t channels, SignalProcessor** processors);
	~MultichannelSignalProcessor();
	virtual void reset();
	virtual void process_buffer(SignalBuffer_t* buffer, size_t channel);
	virtual int has_extra_samples();
	virtual void get_extra_samples(SignalBuffer_t* buffer, size_t channel);
};

