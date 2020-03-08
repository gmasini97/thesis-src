#include "WavOutputWriter.h"


WavOutputWriter::WavOutputWriter(size_t datalen, SF_INFO templ, const char* filename) : OutputWriter(datalen)
{
	this->filename = filename;
	this->info = templ;
	this->file = sf_open(filename, SFM_WRITE, &templ);
	this->real = new float[datalen];
	this->imaginary = new float[datalen];
}
WavOutputWriter::~WavOutputWriter()
{
	sf_close(this->file);
	delete[] this->real;
	delete[] this->imaginary;
}

void WavOutputWriter::reset()
{
}

void WavOutputWriter::process_buffer(SignalBuffer_t* buffer, size_t channel)
{
	size_t size = get_channel_buffer_size(*buffer);
	signal_buffer_to_floats(*buffer, this->real, this->imaginary, channel);
	sf_write_float(this->file, this->real, size);
}