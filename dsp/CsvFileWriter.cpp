#include "CsvFileWriter.h"


CsvFileWriter::CsvFileWriter(AbstractSignalProcessor* p, BitMask channels_to_process, std::string filename) : SignalProcessor(p, channels_to_process)
{
	this->filename = filename;
}
CsvFileWriter::~CsvFileWriter()
{
	fclose(this->file);
}

int CsvFileWriter::init(size_t max_buffer_size, size_t channels)
{
	//TODO MAKE C++ onic;
	this->file = fopen(filename.c_str(), "w");
	if (file == NULL)
		return 0;

	this->count = 0;

	fprintf(this->file, "SMP_N");
	for (size_t channel = 0; channel < channels; channel++)
	{
		if (has_to_process_channel(channel))
		{
			fprintf(this->file, ",Re_%i,Im_%i", (int)channel, (int)channel);
		}
	}
	fprintf(this->file, "\n");

	return SignalProcessor::init(max_buffer_size, channels);
}

void CsvFileWriter::process_buffer(SignalBuffer_t* buffer)
{
	if (has_previous_processor())
		get_previous_processor()->process_buffer(buffer);

	size_t channels = get_channels(*buffer);
	size_t size = get_max_channel_buffer_size(*buffer);
	for (size_t i = 0; i < size; i++)
	{
		fprintf(this->file, "%lli", count++);
		for (size_t channel = 0; channel < channels; channel++)
		{
			if (has_to_process_channel(channel))
			{
				size_t chan_size = get_channel_buffer_size(*buffer, channel);
				if (i < chan_size)
				{
					cuComplex sample = get_signal_buffer_sample(*buffer, channel, i);
					fprintf(this->file, ",%f,%f", sample.x, sample.y);
				}
				else
				{
					fprintf(this->file, ",,");
				}
			}
		}
		fprintf(this->file, "\n");
	}

}