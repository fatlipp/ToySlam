#pragma once

#include <string>
#include <chrono>
#include <unordered_map>

class BlockTimer
{

public:
    BlockTimer(const std::string &caption, const unsigned level = 0);
    ~BlockTimer();

private:
    const std::string caption;
    const unsigned level;

    std::chrono::high_resolution_clock::time_point startTime;
};