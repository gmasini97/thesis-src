#include "pch.h"
#include "FFT.h"

void bit_reversal_sort(float* real, float* imaginary, size_t size)
{
	size_t j, k, halfSize;
	float tmpRe, tmpIm;

	halfSize = size / 2;
	j = halfSize;

	for (size_t i = 1; i < size - 2; i++)
	{
		if (i < j)
		{
			tmpRe = real[j];
			tmpIm = imaginary[j];
			real[j] = real[i];
			imaginary[j] = imaginary[i];
			real[i] = tmpRe;
			imaginary[i] = tmpIm;
		}
		k = halfSize;
		if (k <= j)
		{
			j = j-k;
			k = k/2;
		}
		j = j+k;
	}
}

void butterfly_calculation(float* reA, float* imA, float *reB, float *imB, float reW, float imW)
{
	float reAA = *reA;
	float imAA = *imA;

	float reBW = *reB * reW - *imB * imW;
	float imBW = *reB * imW + *imB * reW;

	*reA += reBW;
	*imA += imBW;
	
	*reB = reAA - reBW;
	*imB = imAA - imBW;
}

void wn(float* re, float* im, size_t exp, size_t size)
{
	*re = cos(2.0 * M_PI * exp / size);
	*im = -sin(2.0 * M_PI * exp / size);
}

void fft(float* real, float* imaginary, size_t size)
{
	float reW, imW;
	size_t levels;
	size_t butterfly_exponent;
	size_t index_a, index_b;

	levels = (size_t)log2(size);

	bit_reversal_sort(real, imaginary, size);

	for (size_t level = 0; level < levels; level++)
	{
		size_t butterflies_per_dft = (size_t)pow(2, level);
		size_t dfts = size / (butterflies_per_dft * 2);

		for (size_t butterfly = 0; butterfly < butterflies_per_dft; butterfly++)
		{
			butterfly_exponent = butterfly * dfts;
			wn(&reW, &imW, butterfly_exponent, size);
			for (size_t dft = 0; dft < dfts; dft++)
			{
				index_a = butterfly + dft * (butterflies_per_dft * 2);
				index_b = index_a + butterflies_per_dft;
				butterfly_calculation(real + index_a, imaginary + index_a, real + index_b, imaginary + index_b, reW, imW);
			}
		}
	}
}

FFTProcessor::FFTProcessor(size_t datalen) : SignalProcessor(datalen)
{
}

void FFTProcessor::reset()
{
}

void FFTProcessor::process_buffer(float* real, float* imaginary, size_t readcount)
{
	fft(real, imaginary, readcount);
}