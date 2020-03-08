#pragma once

#include "SignalProcessor.h"
#include "FFT.h"
#define _USE_MATH_DEFINES
#include <cmath>

void ifft(SignalBuffer_t* buffer, size_t channel);

class IFFTProcessor : public SignalProcessor
{
public:
	IFFTProcessor(size_t datalen);
	~IFFTProcessor();
	void process_buffer(SignalBuffer_t* buffer, size_t channel);
};