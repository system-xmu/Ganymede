#!/bin/bash


set -x

for NR_WARPS in 16; do
  for NR_PAGES in 2048; do
    for NR_ACQUIRE_PAGES in 16; do
      CXXFLAGS="-DNR_WARPS=${NR_WARPS} -DNR_ACQUIRE_PAGES=${NR_ACQUIRE_PAGES} -DNR_PAGES=${NR_PAGES}"
      echo $CXXFLAGS
      cd libgeminiFs_src
      make clean > /dev/null
      make -j CXXFLAGS="${CXXFLAGS}" > /dev/null
      cd ..
      cp libgeminiFs_src/libgeminiFs.a ./build/lib
      
      
      # 3. For examples
      cd examples
      make clean > /dev/null
      make -j CXXFLAGS="${CXXFLAGS}" > /dev/null
#sudo ./TestForNvmeBacking.exe | grep "per s" | awk '{print $4}'
      cd ..
      exit
    done
  done
done
exit


