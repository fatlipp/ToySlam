#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

using TaskType = std::function<void()>;

class ThreadPool 
{
private:
    enum class State
    {
        Stopped,
        Started,
        Kill
    };

public:
    ThreadPool(const unsigned threadsCount);
    ~ThreadPool();

    void AddTask(TaskType&& task);
    void Wait();

private:
    std::vector<std::thread> threads;
    std::queue<TaskType> tasks;

    std::mutex mutex;
    std::condition_variable condition;
    State state;
};
