#!/bin/bash


set -x

for NR_WARPS in 16; do
  for NR_PAGES in 2048; do
    for NR_ACQUIRE_PAGES in 16; do
      CXXFLAGS="-DNR_WARPS=${NR_WARPS} -DNR_ACQUIRE_PAGES=${NR_ACQUIRE_PAGES} -DNR_PAGES=${NR_PAGES}"
      echo $CXXFLAGS
      cd libgeminiFs_src
      make clean
      make -j CXXFLAGS="${CXXFLAGS}"
      cd ..
      cp libgeminiFs_src/libgeminiFs.a ./build/lib
      
      
      # 3. For examples
      cd examples
      make clean
      make TestForNoBackingPagecache.exe -j CXXFLAGS="${CXXFLAGS}"
      ./TestForNoBackingPagecache.exe
      cd ..
      exit
    done
  done
done
exit


