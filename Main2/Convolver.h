#pragma once

#include "SignalProcessor.h"


class Convolver : public SignalProcessor
{
private:
	SignalBuffer_t signal;
	SignalBuffer_t temp;
	size_t temp_index;
	size_t samples_remaining;
public:
	Convolver(size_t datalen, SignalBuffer_t signal);
	~Convolver();
	void reset();
	void process_buffer(SignalBuffer_t* buffer, size_t channel);
	int has_extra_samples();
	void get_extra_samples(SignalBuffer_t* buffer, size_t channel);
};