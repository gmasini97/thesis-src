#pragma once
#include "SignalProcessor.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include "CmplxUtils.cuh"

void dft(SignalBuffer_t* buffer, size_t channel);

class DFTProcessor : public SignalProcessor
{
public:
	DFTProcessor(size_t datalen);
	~DFTProcessor();
	void process_buffer(SignalBuffer_t* buffer, size_t channel);
};