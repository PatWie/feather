#ifndef MISC_H
#define MISC_H

#include <iostream>

namespace feather {
struct cpu {
  static const bool CpuDevice = true;
};

struct gpu {
  static const bool CpuDevice = false;
};


#define runtime_assert(condition, message) (condition                                                     \
                                            ? ((void)0)                                                           \
                                            : ::feather::assertion::detail::assertion_failed_msg(#condition, message, \
                                                                                             __PRETTY_FUNCTION__, __FILE__, __LINE__))

namespace assertion {
namespace detail {

/*!
 * \brief Function call when an assertion failed
 */
template <typename CharT>
void assertion_failed_msg(const CharT* expr, const char* msg, const char* function, const char* file, size_t line) {
  std::cerr
      << "***** Internal Program Error - assertion (" << expr << ") failed in "
      << function << ":\n"
      << file << '(' << line << "): " << msg << std::endl;
  std::abort();
}

} // end of detail namespace
} // end of assertion namespace


} // namespace feather

#endif