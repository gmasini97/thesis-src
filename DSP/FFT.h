#pragma once
#include "SignalProcessor.h"

class FFTProcessor : SignalProcessor
{
private:
	float butterfly_calculation(float* re, float* im);
	void bitreverse_sort(float* data, size_t datalen);
public:
	FFTProcessor();
	void process_buffer(float* data, size_t datalen, int readcount, int channels);
};