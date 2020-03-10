#include "FFT.h"

void bit_reversal_sort(SignalBuffer_t* buffer, size_t channel)
{
	size_t size = get_channel_buffer_size(*buffer, channel);

	cuComplex sample, temporary;

	size_t j, k, halfSize;

	halfSize = size / 2;
	j = halfSize;

	for (size_t i = 1; i < size - 2; i++)
	{
		if (i < j)
		{
			temporary = get_signal_buffer_sample(*buffer, channel, j);
			sample = get_signal_buffer_sample(*buffer, channel, i);
			set_signal_buffer_sample(*buffer, channel, j, sample);
			set_signal_buffer_sample(*buffer, channel, i, temporary);
		}
		k = halfSize;
		while (k <= j)
		{
			j = j-k;
			k = k/2;
		}
		j = j+k;
	}
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
	fft_ws(buffer, channel, size);
}

void fft_ws(SignalBuffer_t* buffer, size_t channel, size_t size)
{

	cuComplex w, wm;

	size_t levels;
	size_t index_a, index_b;

	levels = (size_t)log2(size);

	bit_reversal_sort(buffer, channel);

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
				cuComplex a = get_signal_buffer_sample(*buffer, channel, index_a);
				cuComplex b = get_signal_buffer_sample(*buffer, channel, index_b);
				butterfly_calculation(&a, &b, w);
				set_signal_buffer_sample(*buffer, channel, index_a, a);
				set_signal_buffer_sample(*buffer, channel, index_b, b);
			}
			w = cuCmulf(w, wm);
		}
	}
}

FFTProcessor::FFTProcessor(AbstractSignalProcessor* next, BitMask channels_to_process) : SignalProcessor(next, channels_to_process)
{
}

void FFTProcessor::process_buffer(SignalBuffer_t* buffer)
{
	size_t channels = get_channels(*buffer);
	for (size_t channel = 0; channel < channels; channel++)
	{
		size_t size = get_channel_buffer_size(*buffer, channel);
		if (has_to_process_channel(channel) && size > 0) {
			fft(buffer, channel);
			//set_channel_buffer_size(*buffer, channel, size / 2 + 1);
		}
	}

	if (has_next_processor())
		get_next_processor()->process_buffer(buffer);
}