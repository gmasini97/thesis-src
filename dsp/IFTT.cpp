
#include "IFTT.h"


void idft_wsio(SignalBuffer_t* bufferIn, SignalBuffer_t* bufferOut, size_t channel, size_t size)
{
	cuComplex* tmp = new cuComplex[size];
	cuComplex half_size = make_cuComplex(size/2, 0);
	cuComplex div_two = make_cuComplex(1 / 2, 0);
	cuComplex sample;
	for (size_t i = 0; i < size/2+1; i++)
	{
		sample = get_signal_buffer_sample(*bufferIn, channel, i);
		sample = cuCdivf(sample, half_size);
		set_signal_buffer_sample(*bufferIn, channel, i, cuConjf(sample));
		set_signal_buffer_sample(*bufferOut, channel, i*2+1, make_cuComplex(0,0));
		set_signal_buffer_sample(*bufferOut, channel, i * 2, make_cuComplex(0, 0));
	}

	sample = get_signal_buffer_sample(*bufferIn, channel, 0);
	sample.x /= 2;
	set_signal_buffer_sample(*bufferIn, channel, 0, sample);
	sample = get_signal_buffer_sample(*bufferIn, channel, size/2+1);
	sample.x /= 2;
	set_signal_buffer_sample(*bufferIn, channel, size / 2 + 1, sample);


	for (size_t k = 0; k < size; k++)
	{
		for (size_t i = 0; i < size / 2 + 1; i++) {
			cuComplex out = get_signal_buffer_sample(*bufferOut, channel, i);
			sample = get_signal_buffer_sample(*bufferIn, channel, k);
			cuComplex waves = cuComplex_exp(2 * M_PI * k * i / size);
			out.x += sample.x * waves.x;
			out.y += sample.y * waves.y;
			set_signal_buffer_sample(*bufferOut, channel, i, out);
		}
	}

	delete[] tmp;

}

void ifft(SignalBuffer_t* buffer, size_t channel)
{
	size_t size = get_channel_buffer_size(*buffer, channel);
	ifft_wsio(buffer, buffer, channel, size);
}

void ifft_wsio(SignalBuffer_t* bufferIn, SignalBuffer_t* bufferOut, size_t channel, size_t size)
{
	cuComplex sample;

	for (size_t k = 0; k < size; k++)
	{
		sample = get_signal_buffer_sample(*bufferIn, channel, k);
		sample = cuConjf(sample);
		set_signal_buffer_sample(*bufferOut, channel, k, sample);
	}

	fft_wsio(bufferOut, bufferOut, channel, size);

	cuComplex size_cmplx = make_cuComplex(size, 0);

	for (size_t i = 0; i < size; i++)
	{
		sample = get_signal_buffer_sample(*bufferOut, channel, i);
		sample = cuConjf(cuCdivf(sample, size_cmplx));
		set_signal_buffer_sample(*bufferOut, channel, i, sample);
	}
}

IFFTProcessor::IFFTProcessor(AbstractSignalProcessor* previous, BitMask channels_to_process) : SignalProcessor(previous, channels_to_process)
{
}

void IFFTProcessor::process_buffer(SignalBuffer_t* buffer)
{
	if (has_previous_processor())
		get_previous_processor()->process_buffer(buffer);
	size_t channels = get_channels(*buffer);
	for (size_t channel = 0; channel < channels; channel++)
	{
		size_t size = get_channel_buffer_size(*buffer, channel);
		if (has_to_process_channel(channel) && size > 0) {
			ifft(buffer, channel);
			//set_channel_buffer_size(*buffer, channel, size * 2 - 1);
		}
	}

}