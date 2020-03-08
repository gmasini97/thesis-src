#pragma once
#pragma once

#include "SignalProcessor.h"

class SignalProcessorChain : public SignalProcessor
{
private:
	SignalProcessor** chain;
	size_t chainlen;
public:
	SignalProcessorChain(size_t datalen, SignalProcessor** chain, size_t chainlen);
	~SignalProcessorChain();
	virtual void reset();
	virtual void process_buffer(SignalBuffer_t* buffer, size_t channel);
	virtual int has_extra_samples();
	virtual void get_extra_samples(SignalBuffer_t* buffer, size_t channel);
};