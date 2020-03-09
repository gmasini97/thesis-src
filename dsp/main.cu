#include <iostream>
#include "CsvFileWriter.h"
#include "SndFileLoader.h"
#include "ChainLoader.h"

int main(int argc, const char** argv)
{
	if (argc < 4)
		std::cout << "not enough args" << std::endl;

	size_t buffer_size = atoi(argv[1]);

	AbstractSignalProcessor* chain = build_fx_chain(argv[2]);
	SndFileLoader* fileLoader = new SndFileLoader(chain, BIT_MASK_ALL, argv[3]);

	SF_INFO info = fileLoader->get_info();
	std::cout << "channels " << info.channels << std::endl;
	SignalBuffer_t buffer = create_signal_buffer(buffer_size, info.channels);
	if (!fileLoader->init(buffer_size, info.channels))
	{
		std::cout << "Init fail" << std::endl;
		return 1;
	}

	do
	{
		fileLoader->process_buffer(&buffer);
	} while(get_max_channel_buffer_size(buffer) > 0);

	return 0;
}
