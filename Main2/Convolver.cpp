#include "Convolver.h"
#include <iostream>

size_t bounded_index(SignalBuffer_t buffer, size_t index)
{
	return index >= buffer.maxsize ? index % buffer.maxsize : index;
}

Convolver::Convolver(size_t datalen, SignalBuffer_t signal) : SignalProcessor(datalen)
{
	this->signal = signal;

	this->temp = allocate_signal_buffer(datalen + signal.maxsize - 1);
	this->temp.size = this->temp.maxsize;
	this->temp.channels = 1;

	this->reset();
}

Convolver::~Convolver()
{
	deallocate_signal_buffer(&(this->temp));
}

void Convolver::reset()
{
	for (size_t i = 0; i < this->temp.maxsize; i++)
	{
		set_signal_buffer_sample(this->temp, 0, i, make_cuComplex(0,0));
	}
	this->temp_index = 0;
	this->samples_remaining = 0;
}

void Convolver::process_buffer(SignalBuffer_t* buffer, size_t channel)
{
	size_t buffer_size = get_channel_buffer_size(*buffer);
	size_t signal_size = get_channel_buffer_size(this->signal);
	size_t total = buffer_size + signal_size - 1;

	size_t old_temp_index = this->temp_index;

	for (size_t i = 0; i < buffer_size; i++)
	{
		cuComplex in_sample = get_signal_buffer_sample(*buffer, channel, i);
		for (size_t j = 0; j < signal_size; j++)
		{
			cuComplex signal_sample = get_signal_buffer_sample(this->signal, 0, j);
			size_t index = bounded_index(this->temp, old_temp_index + i + j);
			cuComplex out_sample = get_signal_buffer_sample(this->temp, 0, index);
			//std::cout << out_sample.x << " \t" << out_sample.y << "j" << std::endl;
			cuComplex result = cuCaddf(out_sample, cuCmulf(in_sample, signal_sample));
			set_signal_buffer_sample(this->temp, 0, index, result);
		}
	}

	for (size_t i = 0; i < buffer_size; i++)
	{
		size_t index = bounded_index(this->temp, old_temp_index + i);
		cuComplex out_sample = get_signal_buffer_sample(this->temp, 0, index);
		set_signal_buffer_sample(*buffer, channel, i, out_sample);
		set_signal_buffer_sample(this->temp, 0, index, make_cuComplex(0,0));
	}

	this->temp_index = bounded_index(this->temp, this->temp_index + buffer_size);
	this->samples_remaining = signal_size - 1;
}

int Convolver::has_extra_samples()
{
	return this->samples_remaining > 0;
}

void Convolver::get_extra_samples(SignalBuffer_t* buffer, size_t channel)
{
	size_t max_channel_size = get_max_channel_buffer_size(*buffer);
	size_t current_size = get_channel_buffer_size(*buffer);
	size_t to_copy = fminf(max_channel_size, this->samples_remaining);

	if (to_copy > current_size)
		set_channel_buffer_size(buffer, to_copy);

	for (size_t i = 0; i < to_copy; i++)
	{
		cuComplex sample = get_signal_buffer_sample(this->temp, 0, this->temp_index);
		set_signal_buffer_sample(*buffer, channel, i, sample);
		this->samples_remaining--;
		this->temp_index = bounded_index(this->temp, this->temp_index + 1);
	}
}