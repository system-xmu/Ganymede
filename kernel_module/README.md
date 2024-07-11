# SNVMe
Share the NVMe device between the CPU and GPU.

To realize it, we modified the process of nvme_reset_work to allocated Pages on GPU, and reture its address address to create IO queue;

## IOCTL cmd to set up snvme and create DMA MAP
### NVM_SET_IOQ_NUM
To set the upper limit of IOQs.
### NVM_MAP_HOST_MEMORY and NVM_MAP_DEVICE_MEMORY
To reg a user/device space into snvme, and convert the vaddr to dma addr. the num of map is limited by NVM_SET_IOQ_NUM.
### NVM_SET_SHARE_REG
Set whether the registered map will be used during snvme device registration.
### NVM_UNMAP_MEMORY
To unreg the map on the snvme, the input para include the vaddr of the registered map.

## The Process of user defined IO queue 

create io queue struct, create the map, identify the map attr(sq or cq, and its corresponding sq/cq )
### create IO queue 

### map the userspace 
map_userspace