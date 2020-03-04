#include "DSP.h"
#include "FFT.h"

int main() {
	float f[] = {0, 1, 2, 3, 4, 5, 6, 7};
	FFTProcessor* processor = new FFTProcessor();
	processor->process_buffer(f, 8, 0, 0);

	for (size_t i = 0; i < 8; i++) {
		cout << f[i] << endl;
	}
}