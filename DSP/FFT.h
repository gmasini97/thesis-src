#pragma once
#include "SignalProcessor.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include "CmplxUtils.cuh"

void bit_reversal_sort(SignalBuffer_t* buffer, size_t channel);
void butterfly_calculation(cuComplex* a, cuComplex* b, cuComplex w);
void fft(SignalBuffer_t* buffer, size_t channel);
void fft_ws(SignalBuffer_t* buffer, size_t channel, size_t size);

class FFTProcessor : public SignalProcessor
{
public:
	FFTProcessor(AbstractSignalProcessor* next, BitMask channels_to_process);
	void process_buffer(SignalBuffer_t* buffer);
};