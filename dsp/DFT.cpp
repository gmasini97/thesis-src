#include "DFT.h"

void dft_wsio(SignalBuffer_t* bufferIn, SignalBuffer_t* bufferOut, size_t channel, size_t size)
{
	
	cuComplex* tmp = new cuComplex[size];

	cuComplex sample, s;

	for (size_t k = 0; k < size; k++)
	{
		tmp[k] = make_cuFloatComplex(0,0);
		for (size_t i = 0; i < size; i++)
		{
			s = cuComplex_exp(-2 * M_PI * k * i / size);
			sample = get_signal_buffer_sample(*bufferIn, channel, i);

			tmp[k] = cuCaddf(tmp[k], cuCmulf(sample, s));
		}
	}

	for (size_t k = 0; k < size; k++)
	{
		set_signal_buffer_sample(*bufferOut, channel, k, tmp[k]);
	}

	delete[] tmp;
}

void dft(SignalBuffer_t* buffer, size_t channel)
{
	size_t size = get_channel_buffer_size(*buffer, channel);
	dft_wsio(buffer, buffer, channel, size);
}

DFTProcessor::DFTProcessor(AbstractSignalProcessor* p, BitMask channels_to_process, size_t points) : SignalProcessor(p, channels_to_process)
{
	this->points=points;
}

DFTProcessor::~DFTProcessor()
{
	delete_signal_buffer(avg);
	delete_signal_buffer(tmp);
}

int DFTProcessor::init(size_t max_buffer_size, size_t channels)
{
	size_t size = channels * points;
	size = size > max_buffer_size ? size : max_buffer_size;
	avg = create_signal_buffer(size, channels);
	clear_signal_buffer_deep(avg);
	tmp = create_signal_buffer(size, channels);
	clear_signal_buffer_deep(tmp);
	this->count = 0;
	rem_index = 0;
	samples_remainig = points;
	return SignalProcessor::init(max_buffer_size, channels);
}

void DFTProcessor::process_buffer(SignalBuffer_t* buffer)
{
	if (has_previous_processor()) {
		get_previous_processor()->process_buffer(buffer);

		size_t channels = get_channels(*buffer);
		while (get_max_channel_buffer_size(*buffer) > 0){
			for (size_t c = 0; c < channels; c++) {
				size_t size = get_channel_buffer_size(*buffer, c);
				if (size > 0)
				{
					dft_wsio(buffer, &tmp, c, this->points);
					for (size_t i = 0; i < points; i++) {
						cuComplex s = get_signal_buffer_sample(tmp, c, i);
						cuComplex t = get_signal_buffer_sample(avg, c, i);
						t = cuCmulf(t, make_cuComplex(count, 0));
						t = cuCaddf(t, s);
						t = cuCdivf(t, make_cuComplex(count+1, 0));
						set_signal_buffer_sample(avg,c,i,t);
					}
					count++;
				}
			}
			get_previous_processor()->process_buffer(buffer);
		}

		if (samples_remainig > 0) {
			size_t size = get_max_possible_channel_buffer_size(*buffer, 0);
			for (size_t i = 0; i < size && samples_remainig> 0; i++) {
				for (size_t c = 0; c < channels; c++) {
					cuComplex t = get_signal_buffer_sample(avg, c, rem_index + i);
					set_signal_buffer_sample(*buffer, c, i, t);
				}
				samples_remainig--;
			}
			rem_index+=size;
		}
	}
}