#pragma once

#include "SignalProcessor.h"
#include "FFT.h"
#define _USE_MATH_DEFINES
#include <cmath>

void ifft(SignalBuffer_t* buffer, size_t channel);
void ifft_wsio(SignalBuffer_t* bufferIn, SignalBuffer_t* bufferOut, size_t channel, size_t size);
void idft_wsio(SignalBuffer_t* bufferIn, SignalBuffer_t* bufferOut, size_t channel, size_t size);

class IFFTProcessor : public SignalProcessor
{
public:
	IFFTProcessor(AbstractSignalProcessor* previous, BitMask channels_to_process);
	void process_buffer(SignalBuffer_t* buffer);
};