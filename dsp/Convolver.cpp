#include "Convolver.h"

#define SIGNAL_CHANNEL 0


size_t bounded_index(SignalBuffer_t buffer, size_t channel, size_t index)
{
	size_t channel_size = get_max_possible_channel_buffer_size(buffer, channel);
	return index >= channel_size ? index % channel_size : index;
}





Convolver::Convolver(AbstractSignalProcessor* next, BitMask channels_to_process, SignalBuffer_t signal) : SignalProcessor(next, channels_to_process)
{
	this->signal = signal;
	this->temp = empty_signal_buffer();
}

Convolver::~Convolver()
{
	if(has_next_processor())
		delete get_next_processor();
	delete_signal_buffer(this->temp);
	delete[] this->temp_indexes;
	delete[] this->samples_remaining;
}

int Convolver::init(size_t max_buffer_size, size_t channels)
{
	delete_signal_buffer(this->temp);
	size_t extra_samples = (get_max_buffer_size(this->signal) - 1) * channels;
	this->temp = create_signal_buffer(max_buffer_size + extra_samples, channels);
	clear_signal_buffer_deep(this->temp);

	this->temp_indexes = new size_t[channels]{0};
	this->samples_remaining = new size_t[channels]{0};

	return SignalProcessor::init(max_buffer_size, channels);
}

void Convolver::process_buffer(SignalBuffer_t* buffer)
{
	size_t channels = get_channels(*buffer);

	for (size_t channel = 0; channel < channels; channel++)
	{
		if (!has_to_process_channel(channel))
			continue;

		size_t buffer_size = get_channel_buffer_size(*buffer, channel);
		size_t signal_size = get_channel_buffer_size(this->signal, SIGNAL_CHANNEL);
		size_t total = buffer_size + signal_size - 1;

		size_t temp_index = this->temp_indexes[channel];

		for (size_t i = 0; i < buffer_size; i++)
		{
			cuComplex in_sample = get_signal_buffer_sample(*buffer, channel, i);
			for (size_t j = 0; j < signal_size; j++)
			{
				cuComplex signal_sample = get_signal_buffer_sample(this->signal, SIGNAL_CHANNEL, j);
				size_t index = bounded_index(this->temp, channel, temp_index + i + j);
				cuComplex out_sample = get_signal_buffer_sample(this->temp, channel, index);
				//std::cout << out_sample.x << " \t" << out_sample.y << "j" << std::endl;
				cuComplex result = cuCaddf(out_sample, cuCmulf(in_sample, signal_sample));
				set_signal_buffer_sample(this->temp, channel, index, result);
			}
		}

		for (size_t i = 0; i < buffer_size; i++)
		{
			size_t index = bounded_index(this->temp, channel, temp_index + i);
			cuComplex out_sample = get_signal_buffer_sample(this->temp, channel, index);
			set_signal_buffer_sample(*buffer, channel, i, out_sample);
			set_signal_buffer_sample(this->temp, channel, index, make_cuComplex(0, 0));
		}

		this->temp_indexes[channel] = bounded_index(this->temp, channel, temp_index + buffer_size);
		//this->samples_remaining[channel] = signal_size - 1;
	}

	if (has_next_processor())
		get_next_processor()->process_buffer(buffer);

}








Convolver* create_convolver_from_file(AbstractSignalProcessor* next, BitMask mask, std::string filename, size_t conv_size)
{
	SF_INFO info;
	memset(&info, 0, sizeof(SF_INFO));
	SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &info);

	if (info.channels != 1) {
		std::cout << "only 1 channel convolution kernel allowed" << std::endl;
		return NULL;
	}

	float* real = new float[conv_size];
	float* imag = new float[conv_size] {0};

	size_t actual_read = sf_read_float(file, real, conv_size);

	SignalBuffer_t buffer = create_signal_buffer(conv_size, 1);
	signal_buffer_from_floats(buffer, real, imag, actual_read);

	sf_close(file);
	delete[] real;
	delete[] imag;

	return new Convolver(next, mask, buffer);
}