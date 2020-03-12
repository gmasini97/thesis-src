#pragma once

#include "SignalProcessor.h"
#include <sndfile.h>
#include <stdlib.h>
#include <string>

class SndFileWriter : public SignalProcessor
{
private:
	SNDFILE* file;
	std::string filename;
	SF_INFO info;

	float* reals;
	float* tmp;
public:
	SndFileWriter(AbstractSignalProcessor* p, BitMask channels_to_process, std::string filename, SF_INFO info);
	~SndFileWriter();
	SF_INFO get_info();
	int init(size_t max_buffer_size, size_t channels);
	void process_buffer(SignalBuffer_t* buffer);
};