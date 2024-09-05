#pragma once

#include <iostream>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cstring>

#include <boost/function.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/streambuf.hpp>
#include <boost/asio/write.hpp>
#include <boost/asio/read.hpp>
#include <boost/system/error_code.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <boost/asio/io_context.hpp>