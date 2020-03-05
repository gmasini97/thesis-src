#include "pch.h"
#include "FFT.h"

void bitreverse_sort(float* re, float* im, size_t datalen)
{
	size_t nd2 = datalen / 2;
	size_t j = nd2;
	size_t k = 0;

	float tmp1, tmp2;

	for (size_t i = 1; i < datalen-1; i++)
	{
		if (i < j) {
			tmp1 = re[j];
			tmp2 = im[j];

			re[j] = re[i];
			re[i] = tmp1;

			im[j] = im[i];
			im[i] = tmp2;
		}
		k = nd2;
		while (k <= j) {
			j -= k;
			k /= 2;
		}
		j += k;
	}
}

void butterfly_calculation(float* ar, float* ai, float* br, float* bi, float ur, float ui)
{
	float tr = *br * ur - *bi * ui;
	float ti = *br * ui + *bi * ur;
	*br = *ar - tr;
	*bi = *ai - ti;
	*ar = *ar + tr;
	*ai = *ai + ti;
}

void fft(float* re, float* im, size_t datalen)
{
	size_t m = (size_t)log2(datalen);
	size_t nm1 = datalen - 1;
	size_t le, le2, ip, jm1;
	float ur, ui, sr, si, tr;

	bitreverse_sort(re, im, datalen);

	for (size_t l = 1; l < m; l++) {
		le = (size_t) pow(2, l);
		le2 = le / 2;
		ur = 1;
		ui = 0;
		sr = cos(M_PI / le2);
		si = -sin(M_PI / le2);
		for (size_t j = 1; j < le2; j++) {
			jm1 = j-1;
			for (size_t i = jm1; i < nm1; i += le) {
				ip = i + le2;
				float* ar = re + i;
				float* ai = im + i;
				float* br = re + ip;
				float* bi = im + i;
				butterfly_calculation(ar, ai, br, bi, ur, ui);
			}
			tr = ur;
			ur = tr*sr - ui*si;
			ui = tr*si + ui*sr;
		}
	}
}


FFTProcessor::FFTProcessor(size_t datalen) : SignalProcessor(datalen)
{
	this->im = (float*) malloc(sizeof(float) * datalen);
	this->reset();
}

void FFTProcessor::reset()
{
	size_t datalen = this->getDataLen();
	for (int i = 0; i < datalen; i++) {
		this->im[i] = 0;
	}
}

void FFTProcessor::process_buffer(float* data, size_t readcount)
{
	size_t datalen = this->getDataLen();
	for (size_t i = readcount; i < datalen; i++) {
		data[i] = 0;
	}

	this->re = data;

	fft(this->re, this->im, datalen);

	float xre, xim;

	for (size_t i = 0; i < datalen; i++) {
		xre = this->re[i];
		xim = this->im[i];
		data[i] = sqrt(xre*xre + xim*xim);
	}
};