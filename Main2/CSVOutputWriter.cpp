#include "CSVOutputWriter.h"


CSVOutputWriter::CSVOutputWriter(size_t datalen, const char* filename) : OutputWriter(datalen)
{
	this->filename = filename;
	this->file = fopen(filename, "w");
	this->count = 0;
}
CSVOutputWriter::~CSVOutputWriter()
{
	fclose(this->file);
}

void CSVOutputWriter::reset()
{
	fseek(this->file, 0, SEEK_SET);
	this->count = 0;
}

void CSVOutputWriter::process_buffer(SignalBuffer_t* buffer, size_t channel)
{
	size_t size = get_channel_buffer_size(*buffer);
	cuComplex sample;
	for (size_t i = 0; i < size; i++)
	{
		sample = get_signal_buffer_sample(*buffer, channel, i);
		fprintf(this->file, "%lli,%f,%f\n", count++, sample.x, sample.y);
	}
}