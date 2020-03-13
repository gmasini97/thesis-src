#include "SndFileWriter.h"

SndFileWriter::SndFileWriter(AbstractSignalProcessor* p, BitMask channels_to_process, std::string filename, SF_INFO info) : SignalProcessor(p, channels_to_process)
{
	this->filename = filename;
	this->info = info;
	this->file = sf_open(this->filename.c_str(), SFM_WRITE, &(this->info));
}

SndFileWriter::~SndFileWriter()
{
	sf_close(this->file);
	delete[] this->reals;
	delete[] this->tmp;
}

SF_INFO SndFileWriter::get_info()
{
	return this->info;
}

int SndFileWriter::init(size_t max_buffer_size, size_t channels)
{
	if (this->file == NULL)
		return 0;

	this->reals = new float[max_buffer_size] {0};
	this->tmp = new float[max_buffer_size] {0};

	return SignalProcessor::init(max_buffer_size, channels);
}


void SndFileWriter::process_buffer(SignalBuffer_t* buffer)
{
	if (has_previous_processor())
		get_previous_processor()->process_buffer(buffer);
	size_t channels_to_write = this->info.channels;
	size_t channels = get_channels(*buffer);
	size_t total = signal_buffer_to_floats(*buffer, this->reals, NULL);
	size_t total_per_channel = get_max_channel_buffer_size(*buffer);

	for (size_t i = 0; i < total_per_channel; i++)
	{
		size_t channels_written = 0;
		for (size_t c = 0; c < channels; c++)
		{
			if (!has_to_process_channel(c) || channels_written >= channels_to_write)
				continue;
			float val = this->reals[i * channels + c];
			this->tmp[i * channels_to_write + channels_written] = val;
			channels_written++;
		}
		while (channels_written < channels_to_write) {
			this->tmp[i * channels_to_write + channels_written] = 0;
			channels_written++;
		}
	}

	sf_write_float(this->file, this->tmp, total_per_channel * channels_to_write);

}