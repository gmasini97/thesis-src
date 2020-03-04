#pragma once

#include "DSP.h"

namespace DSP {
	class LPFilterAvg : SignalProcessor
	{
	public:
		LPFilterAvg(float cutoffFrequency);
		void process_buffer(float* data, int readcount, int channels);
	};
}