#include "ToPolar.h"

cuComplex polar(cuComplex* x)
{
	float re = x->x;
	float im = x->y;

	float magnitude = sqrt(re*re + im*im);
	float phase = atan2(im, re);

	return make_cuComplex(magnitude, phase);
}

ToPolar::ToPolar(size_t datalen) : SignalProcessor(datalen)
{
}
ToPolar::~ToPolar()
{
}

void ToPolar::process_buffer(SignalBuffer_t* buffer, size_t channel)
{
	size_t size = get_channel_buffer_size(*buffer);
	cuComplex sample;
	for (size_t i = 0; i < size; i++)
	{
		sample = get_signal_buffer_sample(*buffer, channel, i);
		set_signal_buffer_sample(*buffer, channel, i, polar(&sample));
	}
}