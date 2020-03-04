#include "pch.h"
#include "FFT.h"

FFTProcessor::FFTProcessor()
{

}

void FFTProcessor::bitreverse_sort(float* data, size_t datalen)
{
	size_t nd2 = datalen / 2;
	size_t j = nd2;
	size_t k = 0;

	float tmp;

	for (size_t i = 1; i < datalen-1; i++)
	{
		if (i < j) {
			tmp = data[j];
			data[j] = data[i];
			data[i] = tmp;
		}
		k = nd2;
		while (k <= j) {
			j -= k;
			k /= 2;
		}
		j += k;
	}
}

void FFTProcessor::process_buffer(float* data, size_t datalen, int readcount, int channels)
{
	this->bitreverse_sort(data, datalen);
}