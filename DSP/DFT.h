#pragma once
#include "SignalProcessor.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include "CmplxUtils.cuh"

void dft(SignalBuffer_t* buffer, size_t channel);
void dft_wsio(SignalBuffer_t* bufferIn, SignalBuffer_t* bufferOut, size_t channel, size_t size);

class DFTProcessor : public SignalProcessor
{
private:
	size_t points;
public:
	DFTProcessor(AbstractSignalProcessor* p, BitMask channels_to_process, size_t points);
	~DFTProcessor();
	void process_buffer(SignalBuffer_t* buffer);
};