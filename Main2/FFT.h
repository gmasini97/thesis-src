#pragma once
#include "SignalProcessor.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include "CmplxUtils.cuh"

void bit_reversal_sort(SignalBuffer_t* buffer, size_t channel);
void butterfly_calculation(cuComplex* a, cuComplex* b, cuComplex w);
void fft(SignalBuffer_t* buffer, size_t channel);

class FFTProcessor : public SignalProcessor
{
public:
	FFTProcessor(size_t datalen);
	~FFTProcessor();
	void process_buffer(SignalBuffer_t* buffer, size_t channel);
};