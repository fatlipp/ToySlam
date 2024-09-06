#include "ThreadPool.h"

#include <iostream>

ThreadPool::ThreadPool(const unsigned threadsCount)
    : state { State::Stopped }
{
    // each of the threads will wait for a task or a Kill command
    for (unsigned i = 0; i < threadsCount; ++i)
    {
        threads.emplace_back([this] {
            while (true)
            {
                TaskType currentTask;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    condition.wait(lock, [this] { 
                            return !tasks.empty() || state != State::Stopped;
                        });

                    if (state == State::Kill)
                    {
                        break;
                    }

                    if (tasks.empty())
                    {
                        state = State::Stopped;
                        condition.notify_all();
                        continue;
                    }

                    currentTask = std::move(tasks.front());
                    tasks.pop();
                }

                if (currentTask)
                {
                    currentTask();
                }
            }
        });
    }
}

ThreadPool::~ThreadPool()
{
    {
        std::lock_guard<std::mutex> lock { mutex };
        if (state == State::Kill)
        {
            return;
        }
        state = State::Kill;
    }
    condition.notify_all();

    for (auto& t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }
}

void ThreadPool::AddTask(TaskType&& task)
{
    {
        std::lock_guard<std::mutex> lock(mutex);
        tasks.emplace(std::move(task));
        state = State::Started;
    }
    condition.notify_one();
}

void ThreadPool::Wait()
{
    {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [this] { 
                return tasks.empty() && state == State::Stopped;
            });
    }
}