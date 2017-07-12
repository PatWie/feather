#ifndef MISC_H
#define MISC_H

namespace feather {
  struct cpu
  {
    static const bool CpuDevice = true;
  };

  struct gpu
  {
    static const bool CpuDevice = false;
  };
}

#endif