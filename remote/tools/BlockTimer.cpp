#include "tools/BlockTimer.h"

#include <iostream>
#include <iomanip>

BlockTimer::BlockTimer(const std::string &caption, const unsigned level)
    : caption(caption)
    , level(level)
{
    this->startTime = std::chrono::high_resolution_clock::now();
}

BlockTimer::~BlockTimer()
{
    const auto endTime = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - this->startTime).count();

    std::cout << std::setw(level) << "" << "[" << caption << "] time: " << duration << "ms" << std::endl;
}