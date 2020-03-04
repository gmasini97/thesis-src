#pragma once

class SignalProcessor {
public:
	virtual void process_buffer(float* data, size_t datalen, int readcount, int channels) = 0;
};
