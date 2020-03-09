#include "SndFileLoader.h"


SndFileLoader::SndFileLoader(AbstractSignalProcessor* p, BitMask channels_to_process, const char* filename) : SignalProcessor(p, channels_to_process)
{
	this->filename = filename;
	this->file = sf_open(this->filename, SFM_READ, &(this->info));
}

SndFileLoader::~SndFileLoader()
{
	sf_close(this->file);
	delete[] this->reals;
	delete[] this->imags;
}

SF_INFO SndFileLoader::get_info()
{
	return this->info;
}

int SndFileLoader::init(size_t max_buffer_size, size_t channels)
{
	if (this->file == NULL)
		return 0;

	this->reals = new float[max_buffer_size]{0};
	this->imags = new float[max_buffer_size]{0};

	return SignalProcessor::init(max_buffer_size, channels);
}


void SndFileLoader::process_buffer(SignalBuffer_t* buffer)
{
	size_t channels = get_channels(*buffer);
	size_t max_size = get_max_buffer_size(*buffer);
	size_t read_count = sf_read_float(this->file, reals, max_size);
	signal_buffer_from_floats(*buffer, this->reals, this->imags, read_count);
	
	if (has_next_processor())
		get_next_processor()->process_buffer(buffer);
}