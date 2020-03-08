#pragma once
#include "SignalProcessor.h"

class OutputWriter : public SignalProcessor
{
public:
	OutputWriter(size_t datalen);
	~OutputWriter();
};