#pragma once

#include "SignalProcessor.h"
#define _USE_MATH_DEFINES
#include <cmath>

class GainProcessor : public SignalProcessor
{
private:
	cuComplex gain;
public:
	GainProcessor(AbstractSignalProcessor* previous, BitMask channels_to_process, float re_gain, float im_gain);
	void process_buffer(SignalBuffer_t* buffer);
};