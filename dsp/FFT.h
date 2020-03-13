#pragma once
#include "SignalProcessor.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include "CmplxUtils.cuh"

void bit_reversal_sort(SignalBuffer_t* buffer, size_t channel);
void bit_reversal_sort_wsio(SignalBuffer_t* bufferIn, SignalBuffer_t* bufferOut, size_t channel, size_t size);
void butterfly_calculation(cuComplex* a, cuComplex* b, cuComplex w);
void fft(SignalBuffer_t* buffer, size_t channel);
void fft_wsio(SignalBuffer_t* bufferIn, SignalBuffer_t* bufferOut, size_t channel, size_t size);

class FFTProcessor : public SignalProcessor
{
private:
	size_t points;
public:
	FFTProcessor(AbstractSignalProcessor* previous, BitMask channels_to_process, size_t points);
	void process_buffer(SignalBuffer_t* buffer);
};