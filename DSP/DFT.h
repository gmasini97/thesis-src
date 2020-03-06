#pragma once
#include "SignalProcessor.h"
#define _USE_MATH_DEFINES
#include <cmath>

void dft(float* real, float* imaginary, size_t size);

class DFTProcessor : public SignalProcessor
{
public:
	DFTProcessor(size_t datalen);
	void reset();
	void process_buffer(float* real, float* imaginary, size_t readcount);
};