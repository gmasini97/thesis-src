#pragma once
#include "SignalProcessor.h"
#define _USE_MATH_DEFINES
#include <cmath>

void bit_reversal_sort(float* real, float* imaginary, size_t size);
void butterfly_calculation(float* reA, float* imA, float* reB, float* imB, float reW, float imW);
void fft(float* real, float* imaginary, size_t size);

class FFTProcessor : public SignalProcessor
{
public:
	FFTProcessor(size_t datalen);
	~FFTProcessor();
	void reset();
	void process_buffer(float* real, float* imaginary, size_t readcount);
};