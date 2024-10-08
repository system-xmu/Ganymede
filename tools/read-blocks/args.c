#include "args.h"
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <fcntl.h>
#include <limits.h>
#include <errno.h>
#include <string.h>



static struct option opts[] = {
    { .name = "help", .has_arg = no_argument, .flag = NULL, .val = 'h' },
    { .name = "ctrl", .has_arg = required_argument, .flag = NULL, .val = 'c' },
    { .name = "namespace", .has_arg = required_argument, .flag = NULL, .val = 'n' },
    { .name = "ns", .has_arg = required_argument, .flag = NULL, .val = 'n' },
    { .name = "blocks", .has_arg = required_argument, .flag = NULL, .val = 'b' },
    { .name = "offset", .has_arg = required_argument, .flag = NULL, .val = 'o' },
    { .name = "output", .has_arg = required_argument, .flag = NULL, .val = 0 },
    { .name = "ascii", .has_arg = no_argument, .flag = NULL, .val = 2 },
    { .name = "identify", .has_arg = no_argument, .flag = NULL, .val = 3 },
    { .name = "chunk", .has_arg = required_argument, .flag = NULL, .val = 's' },
    { .name = "depth", .has_arg = required_argument, .flag = NULL, .val = 'q' },
    { .name = "write", .has_arg = required_argument, .flag = NULL, .val = 'w' },
    { .name = NULL, .has_arg = no_argument, .flag = NULL, .val = 0 }
};



static void show_usage(const char* name)
{
    fprintf(stderr, "Usage: %s --ctrl <path> --blocks <count> [--offset <count>] [--ns <id>] [--ascii | --output <path>]\n", name);
}



static void show_help(const char* name)
{
    show_usage(name);

    fprintf(stderr, ""
            "    --ctrl         <path>    Specify path to controller.\n"
            "    --chunk        <count>   Limit reads to a number of blocks at the time.\n"
            "    --depth        <count>   Set submission queue depth.\n"
            "    --blocks       <count>   Read specified number of blocks from disk.\n"
            "    --offset       <count>   Start reading at specified block (default 0).\n"
            "    --namespace    <id>      Namespace identifier (default 1).\n"
            "    --ascii                  Show output of ASCII characters as text.\n"
            "    --output       <path>    Dump to file rather than stdout.\n"
            "    --write        <path>    Read file and write to disk before reading back.\n"
            "    --identify               Show IDENTIFY CONTROLLER structure.\n"
           );
}



void parse_options(int argc, char** argv, struct options* args)
{

    const char* argstr = ":hc:b:n:o:s:w:q:";
    int opt;
    int idx;
    char* endptr;

    args->controller_path = NULL;
    args->queue_size = 0;
    args->chunk_size = 0;
    args->namespace_id = 1;
    args->num_blocks = 0;
    args->offset = 0;
    args->output = NULL;
    args->input = NULL;
    args->ascii = false;
    args->identify = false;

    while ((opt = getopt_long(argc, argv, argstr, opts, &idx)) != -1)
    {
        switch (opt)
        {
            case '?':
                fprintf(stderr, "Unknown option: `%s'\n", argv[optind - 1]);
                exit('?');

            case ':':
                fprintf(stderr, "Missing argument for option %s\n", argv[optind - 1]);
                exit('?');

            case 'h':
                show_help(argv[0]);
                exit('?');

            case 0:
                if (args->ascii)
                {
                    fprintf(stderr, "Output file is set, ignoring option --ascii\n");
                    args->ascii = false;
                }

                args->output = fopen(optarg, "wb");
                if (args->output == NULL)
                {
                    fprintf(stderr, "Failed to open output file: %s\n", strerror(errno));
                    exit(1);
                }
                break;

            case 'w':
                args->input = fopen(optarg, "rb");
                if (args->input == NULL)
                {
                    fprintf(stderr, "Failed to open input file: %s\n", strerror(errno));
                    exit(1);
                }
                break;


            case 3:
                args->identify = true;
                break;


            case 'c':
                args->controller_path = optarg;
                break;
            case 'n':
                args->namespace_id = strtoul(optarg, &endptr, 0);
                if (endptr == NULL || *endptr != '\0' || args->namespace_id == 0xffffffff)
                {
                    fprintf(stderr, "Invalid namespace identifier: `%s'\n", optarg);
                    exit(2);
                }
                break;

            case 2:
                if (args->output == NULL)
                {
                    args->ascii = true;
                }
                else
                {
                    fprintf(stderr, "Output file is set, ignoring option %s\n", argv[optind - 1]);
                }
                break;

            case 'o':
                args->offset = strtoul(optarg, &endptr, 0);
                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid block count: `%s'\n", optarg);
                    exit(2);
                }
                break;

            case 'b':
                args->num_blocks = strtoul(optarg, &endptr, 0);
                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid block count: `%s'\n", optarg);
                    exit(2);
                }

                if (args->num_blocks == 0)
                {
                    fprintf(stderr, "Number of blocks can not be 0!\n");
                    exit(2);
                }
                break;

            case 's':
                args->chunk_size = strtoul(optarg, &endptr, 0);
                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid block count: `%s'\n", optarg);
                    exit(2);
                }
                break;

            case 'q':
                args->queue_size = strtoul(optarg, &endptr, 0);
                if (endptr == NULL || *endptr != '\0')
                {
                    fprintf(stderr, "Invalid queue depth: `%s'\n", optarg);
                    exit(2);
                }
                break;
        }
    }


    if (args->controller_path == NULL)
    {
        fprintf(stderr, "No controller specified!\n");
        show_usage(argv[0]);
        exit(1);
    }

    if (args->num_blocks == 0)
    {
        fprintf(stderr, "Block count is not specified!\n");
        show_usage(argv[0]);
        exit(2);
    }

    if (args->chunk_size == 0)
    {
        args->chunk_size = args->num_blocks;
    }

    if (args->queue_size == 0)
    {
        args->queue_size = 64;
    }
}

