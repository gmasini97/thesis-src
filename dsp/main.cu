#include <iostream>
#include "CsvFileWriter.h"
#include "SndFileLoader.h"
#include "ChainLoader.h"

int main(int argc, const char** argv)
{
	if (argc < 4)
		std::cout << "not enough args" << std::endl;

	size_t buffer_size = atoi(argv[1]);

	SndFileLoader* fileLoader = new SndFileLoader(NULL, BIT_MASK_ALL, argv[3]);
	AbstractSignalProcessor* chain = build_fx_chain(argv[2], fileLoader);

	SF_INFO info = fileLoader->get_info();
	std::cout << "channels " << info.channels << std::endl;
	SignalBuffer_t buffer = create_signal_buffer(buffer_size, info.channels);
	if (!chain->init(buffer_size, info.channels))
	{
		std::cout << "Init fail" << std::endl;
		return 1;
	}

	auto start = std::chrono::high_resolution_clock::now();
	do
	{
		chain->process_buffer(&buffer);
	} while(get_max_channel_buffer_size(buffer) > 0);
	auto stop = std::chrono::high_resolution_clock::now();

	long long int time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

	printf("%lli\n", time);

	return 0;
}
