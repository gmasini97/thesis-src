#include "FFT.h"

void bit_reversal_sort_wsio(SignalBuffer_t* bufferIn, SignalBuffer_t* bufferOut, size_t channel, size_t size)
{
	cuComplex sample, temporary;
	size_t buffer_size = get_channel_buffer_size(*bufferIn, channel);
	size_t j, k, halfSize;

	halfSize = size / 2;
	j = halfSize;

	sample = get_signal_buffer_sample(*bufferIn, channel, 0);
	set_signal_buffer_sample(*bufferOut, channel, 0, sample);

	sample = get_signal_buffer_sample(*bufferIn, channel, size-1);
	set_signal_buffer_sample(*bufferOut, channel, size - 1, sample);

	for (size_t i = 1; i < size - 2; i++)
	{
		if (i < j)
		{
			temporary = get_signal_buffer_sample(*bufferIn, channel, j);
			sample = get_signal_buffer_sample(*bufferIn, channel, i);

			if (i >= buffer_size)
				sample = make_cuComplex(0,0);
			if (j >= buffer_size)
				temporary = make_cuComplex(0, 0);

			set_signal_buffer_sample(*bufferOut, channel, j, sample);
			set_signal_buffer_sample(*bufferOut, channel, i, temporary);
		}
		else if (i == j) {
			sample = get_signal_buffer_sample(*bufferIn, channel, i);
			if (i >= buffer_size)
				sample = make_cuComplex(0, 0);
			set_signal_buffer_sample(*bufferOut, channel, i, sample);
		}
		k = halfSize;
		while (k <= j)
		{
			j = j - k;
			k = k / 2;
		}
		j = j + k;
	}
}

void bit_reversal_sort(SignalBuffer_t* buffer, size_t channel)
{
	size_t size = get_channel_buffer_size(*buffer, channel);
	bit_reversal_sort_wsio(buffer, buffer, channel, size);
}

void butterfly_calculation(cuComplex* a, cuComplex* b, cuComplex w)
{
	cuComplex aa = *a;
	cuComplex bw = cuCmulf(*b, w);

	*a = cuCaddf(aa, bw);
	*b = cuCsubf(aa, bw);
}

void fft(SignalBuffer_t* buffer, size_t channel)
{
	size_t size = get_channel_buffer_size(*buffer, channel);
	fft_wsio(buffer, buffer, channel, size);
}

void fft_wsio(SignalBuffer_t* bufferIn, SignalBuffer_t* bufferOut, size_t channel, size_t size_in)
{

	cuComplex w, wm;

	size_t levels;
	size_t index_a, index_b;

	size_t size = (size_t)pow(2, ceil(log2(size_in)));;

	levels = (size_t)log2(size);

	for (int i = 0; i < 8; i++) {
		LOG("%.3f  ", get_signal_buffer_sample(*bufferOut,channel,i).x);
	}
	LOG("\n");
	bit_reversal_sort_wsio(bufferIn, bufferOut, channel, size);
	for (int i = 0; i < 8; i++) {
		LOG("%.3f  ", get_signal_buffer_sample(*bufferOut, channel, i).x);
	}
	LOG("\n");

	for (size_t level = 0; level < levels; level++)
	{
		size_t butterflies_per_dft = (size_t)pow(2, level);
		size_t dfts = size / (butterflies_per_dft * 2);

		wm = cuComplex_exp(-(M_PI / butterflies_per_dft));
		w = make_cuComplex(1,0);
		for (size_t butterfly = 0; butterfly < butterflies_per_dft; butterfly++)
		{
			for (size_t dft = 0; dft < dfts; dft++)
			{
				index_a = butterfly + dft * (butterflies_per_dft * 2);
				index_b = index_a + butterflies_per_dft;
				cuComplex a = get_signal_buffer_sample(*bufferOut, channel, index_a);
				cuComplex b = get_signal_buffer_sample(*bufferOut, channel, index_b);
				butterfly_calculation(&a, &b, w);
				set_signal_buffer_sample(*bufferOut, channel, index_a, a);
				set_signal_buffer_sample(*bufferOut, channel, index_b, b);
			}
			w = cuCmulf(w, wm);
		}
	}
}

FFTProcessor::FFTProcessor(AbstractSignalProcessor* next, BitMask channels_to_process, size_t points) : SignalProcessor(next, channels_to_process)
{
	this->points = points;
}

void FFTProcessor::process_buffer(SignalBuffer_t* buffer)
{
	size_t channels = get_channels(*buffer);
	for (size_t channel = 0; channel < channels; channel++)
	{
		size_t size = get_channel_buffer_size(*buffer, channel);
		if (has_to_process_channel(channel) && size > 0) {
			fft_wsio(buffer, buffer, channel, points);
			//set_channel_buffer_size(*buffer, channel, size / 2 + 1);
		}
	}

	if (has_next_processor())
		get_next_processor()->process_buffer(buffer);
}