#pragma once

#include "SignalProcessor.h"
#include <sndfile.h>
#include <stdlib.h>

class SndFileLoader : public SignalProcessor
{
private:
	SNDFILE* file;
	const char* filename;
	SF_INFO info;

	float* reals;
	float* imags;
public:
	SndFileLoader(AbstractSignalProcessor* p, BitMask channels_to_process, const char* filename);
	~SndFileLoader();
	SF_INFO get_info();
	int init(size_t max_buffer_size, size_t channels);
	void process_buffer(SignalBuffer_t* buffer);
};