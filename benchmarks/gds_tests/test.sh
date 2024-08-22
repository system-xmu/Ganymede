#!/bin/bash -x
make clean; make
THREADS=(1 4  8 16 32 64 128 256 512)  
for threads in "${THREADS[@]}"  
do  
    logfile="test_gds_cuFileRead_latency_threads_${threads}.log"  
    
    rm "$logfile"

    ./test_gds_latency 1 "$threads" ~/nvme0n1_geminifs/test_file.txt >> "$logfile"
    

    echo "Run test_gds_cuFileRead success: threads=${threads}"  
done  

