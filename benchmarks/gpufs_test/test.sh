#!/bin/bash -x
make clean; make
BLOCKS=(1)  
THREADS=( 16 32 64 128 256 512 1024)  
for blocks in "${BLOCKS[@]}"  
do  
    for threads in "${THREADS[@]}"  
    do  
        logfile="test_gpufs_read_latency_4M_b_${blocks}_t_${threads}.log"  
        
        rm "$logfile"
        # make sure generate 500,0000 i/o
        if [ "$threads" -eq 256 ]; then
            ./test_latency_bak 2 "$blocks" "$threads" ~/nvme0n1_geminifs/test_file.txt >> "$logfile"
        elif [ "$threads" -eq 512 ] ; then
            ./test_latency_bak 40 "$blocks" "$threads" ~/nvme0n1_geminifs/test_file.txt >> "$logfile"
        elif [ "$threads" -eq 1024 ]; then
            ./test_latency_bak 80 "$blocks" "$threads" ~/nvme0n1_geminifs/test_file.txt >> "$logfile"
        else
            ./test_latency_bak 1 "$blocks" "$threads" ~/nvme0n1_geminifs/test_file.txt >> "$logfile"
        fi
        echo "Run test_gpufs_read_latency success: blocks=${blocks}, threads=${threads}"  
    done  
done
