#ifndef __LIBNVM_SAMPLES_READ_BLOCKS_OPTIONS_H__
#define __LIBNVM_SAMPLES_READ_BLOCKS_OPTIONS_H__

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <nvm_types.h>


struct options
{

    const char* controller_path;
    size_t      queue_size;
    size_t      chunk_size;
    uint32_t    namespace_id;
    size_t      num_blocks;
    size_t      offset;
    FILE*       output;
    FILE*       input;
    bool        ascii;
    bool        identify;
};


void parse_options(int argc, char** argv, struct options* options);


#endif
