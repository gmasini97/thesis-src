#pragma once

#include "DSP.h"

namespace DSP {
	class LPFilterAvg : SignalProcessor
	{
	private:
		size_t points;
	public:
		LPFilterAvg(size_t datalen, size_t points);
		void reset();
		void process_buffer(float* data, int readcount, int channels);
	};
}