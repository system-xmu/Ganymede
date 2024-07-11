#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_dma.h>
#include <nvm_aq.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <getopt.h>
#include <string.h>
#include <errno.h>

#ifdef __DIS_CLUSTER__
#include <sisci_api.h>
#include <dis/dis_types.h>
#define MAX_ADAPTERS DIS_MAX_NSCIS
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include "integrity.h"

#define snvme_control_path "/dev/snvm_control"
#define snvme_path "/dev/snvme1"


struct arguments
{
    uint64_t        device_id;
    const char*     device_path;
    uint32_t        ns_id;
    uint16_t        n_queues;
    uint64_t        read_bytes;
    const char*     filename;
    const char*     device_control_path;
    
};



static bool parse_number(uint64_t* number, const char* str, int base, uint64_t lo, uint64_t hi)
{
    char* endptr = NULL;
    uint64_t ul = strtoul(str, &endptr, base);

    if (endptr == NULL || *endptr != '\0')
    {
        return false;
    }

    if (lo < hi && (ul < lo || ul >= hi))
    {
        return false;
    }

    *number = ul;
    return true;
}



static void parse_arguments(int argc, char** argv, struct arguments* args)
{
    struct option opts[] = {
        { "help", no_argument, NULL, 'h' },
        { "read", required_argument, NULL, 'r' },
        { "ctrl", required_argument, NULL, 'c' },
        { "namespace", required_argument, NULL, 'n' },
        { "queues", required_argument, NULL, 'q' },
        { "device_control_path", required_argument, NULL, 'p' },
        { "device_chr_path", required_argument, NULL, 'd' },
        { NULL, 0, NULL, 0 }
    };

    int opt;
    int idx;
    uint64_t num;

    args->device_id = 0;
    args->device_path = NULL;
    args->ns_id = 1;
    args->n_queues = 1;
    args->read_bytes = 0;
    args->filename = NULL;
    args->device_control_path = NULL;
  

    while ((opt = getopt_long(argc, argv, ":hr:c:n:q:p:d", opts, &idx)) != -1)
    {
        switch (opt)
        {
            case '?': // unknown option
            default:
                fprintf(stderr, "Unknown option: `%s'\n", argv[optind - 1]);
                exit(4);

            case ':': // missing option argument
                fprintf(stderr, "Missing argument for option `%s'\n", argv[optind - 1]);
                exit(4);

            case 'h':
                fprintf(stderr, "Usage: %s --ctrl=device-path [--read=bytes] [-n namespace] [-q queues] filename\n", argv[0]);
                exit(4);

            case 'r':
                if ( !parse_number(&args->read_bytes, optarg, 0, 0, 0) || args->read_bytes == 0 )
                {
                    fprintf(stderr, "Invalid number of bytes: `%s'\n", optarg);
                    exit(1);
                }
                break;

            case 'c': // device identifier
                args->device_path = optarg;
                break;
    
            case 'n': // specify namespace number
                if (! parse_number(&num, optarg, 0, 0, 0) )
                {
                    fprintf(stderr, "Invalid controller identifier: `%s'\n", optarg);
                    exit(3);
                }
                args->ns_id = (uint32_t) num;
                break;

            case 'q': // set number of queues
                if (! parse_number(&num, optarg, 0, 1, 0xffff) )
                {
                    fprintf(stderr, "Invalid number of queues: `%s'\n", optarg);
                    exit(3);
                }
                args->n_queues = (uint16_t) num;
                break;
        }
    }

    argc -= optind;
    argv += optind;


    if (args->device_path == NULL)
    {
        fprintf(stderr, "No controller specified!\n");
        exit(1);
    }


    if (argc < 1)
    {
        fprintf(stderr, "File not specified!\n");
        exit(2);
    }
    else if (argc > 1)
    {
        fprintf(stderr, "More than one filename specified!\n");
        exit(2);
    }

    args->filename = argv[0];
}


static void print_ctrl_info(FILE* fp, const struct nvm_ctrl_info* info)
{
    unsigned char vendor[4];
    memcpy(vendor, &info->pci_vendor, sizeof(vendor));

    char serial[21];
    memset(serial, 0, 21);
    memcpy(serial, info->serial_no, 20);

    char model[41];
    memset(model, 0, 41);
    memcpy(model, info->model_no, 40);

    char revision[9];
    memset(revision, 0, 9);
    memcpy(revision, info->firmware, 8);

    fprintf(fp, "------------- Controller information -------------\n");
    fprintf(fp, "PCI Vendor ID           : %x %x\n", vendor[0], vendor[1]);
    fprintf(fp, "PCI Subsystem Vendor ID : %x %x\n", vendor[2], vendor[3]);
    fprintf(fp, "NVM Express version     : %u.%u.%u\n",
            info->nvme_version >> 16, (info->nvme_version >> 8) & 0xff, info->nvme_version & 0xff);
    fprintf(fp, "Controller page size    : %zu\n", info->page_size);
    fprintf(fp, "Max queue entries       : %u\n", info->max_entries);
    fprintf(fp, "Serial Number           : %s\n", serial);
    fprintf(fp, "Model Number            : %s\n", model);
    fprintf(fp, "Firmware revision       : %s\n", revision);
    fprintf(fp, "Max data transfer size  : %zu\n", info->max_data_size);
    fprintf(fp, "Max outstanding commands: %zu\n", info->max_out_cmds);
    fprintf(fp, "Max number of namespaces: %zu\n", info->max_n_ns);
    fprintf(fp, "--------------------------------------------------\n");
}







static void remove_queues(struct queue* queues, uint16_t n_queues)
{
    uint16_t i;

    if (queues != NULL)
    {

        for (i = 0; i < n_queues; ++i)
        {
            remove_queue(&queues[i]);
        }

        free(queues);
    }
}



static int request_queues(nvm_ctrl_t* ctrl, struct queue** queues)
{
    struct queue* q;
    *queues = NULL;
    uint16_t i;
    int status;

    status = ioctl_set_qnum(ctrl, ctrl->cq_num+ctrl->sq_num);
    if (status != 0)
    {
    
        return status;
    }
    // Allocate queue descriptors
    q = (queue *)calloc(ctrl->cq_num+ctrl->sq_num, sizeof(struct queue));
    if (q == NULL)
    {
        fprintf(stderr, "Failed to allocate queues: %s\n", strerror(errno));
        return ENOMEM;
    }

    // Create completion queue
    for (i = 0; i < ctrl->cq_num; ++i)
    {
        status = create_queue(&q[i], ctrl, NULL, i);
        if (status != 0)
        {
            free(q);
            return status;
        }
    }


    // Create submission queues
    for (i = 0; i < ctrl->sq_num; ++i)
    {
        status = create_queue(&q[i + ctrl->cq_num], ctrl, &q[0], i+ctrl->cq_num);
        if (status != 0)
        {
            remove_queues(q, i);
            return status;
        }
    }
    
    *queues = q;
    return status;
}



int main(int argc, char** argv)
{

    
    nvm_ctrl_t* ctrl;
    int status;
    nvm_dma_t* aq_dma;
    struct disk disk;
    struct queue* queues = NULL;
    struct buffer buffer;
    int snvme_c_fd,snvme_d_fd;
    // Parse command line arguments
     
    int ioq_num;
    int read_bytes;
    ioq_num = 1;
    read_bytes = 1024*1024*1;
    snvme_c_fd = open(snvme_control_path, O_RDWR | O_NONBLOCK);
    if (snvme_c_fd < 0)
    {

        fprintf(stderr, "Failed to open device file: %s\n", strerror(errno));
        exit(1);
    }

    // Get controller reference

    snvme_d_fd = open(snvme_path, O_RDWR | O_NONBLOCK);
    if (snvme_d_fd < 0)
    {
        fprintf(stderr, "Failed to open device file: %s\n", strerror(errno));
        exit(1);
    }

    status = nvm_ctrl_init(&ctrl, snvme_c_fd, snvme_d_fd);
    if (status != 0)
    {
        close(snvme_c_fd);
        close(snvme_d_fd);
        
        fprintf(stderr, "Failed to get controller reference: %s\n", strerror(status));
    }
    
    close(snvme_c_fd);
    close(snvme_d_fd);

    ctrl->cq_num = 1;
    ctrl->sq_num = 4;
    
    // Create queues
    status = request_queues(ctrl, &ctrl->queues);
    if (status != 0)
    {
        goto out;
    }

    fprintf(stderr, "Using %u submission queues:\n", ioq_num);

    // if (read_bytes > 0)
    // {
    //     status = disk_read(&disk, &buffer, queues, ioq_num, fp, file_size);
    // }
    // else
    // {
    //     status = disk_write(&disk, &buffer, queues, ioq_num, fp, file_size);
    // }

out:
    remove_queues(ctrl->queues, ctrl->cq_num+ctrl->sq_num);
    //remove_buffer(&buffer);
    nvm_ctrl_free(ctrl);
    exit(status);
}
