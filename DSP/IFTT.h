#pragma once

#include "SignalProcessor.h"
#include "FFT.h"
#define _USE_MATH_DEFINES
#include <cmath>


class IFFTProcessor : public SignalProcessor
{
public:
	IFFTProcessor(size_t datalen);
	~IFFTProcessor();
	void reset();
	void process_buffer(float* real, float* imaginary, size_t readcount);
};