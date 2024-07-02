#ifndef _SNVME_HELP_H
#define _SNVME_HELP_H




struct nds_mapping {
    // input 
    int file_fd;  
	loff_t offset;  /* file offset */
	__u64 len;
    // output 
    __u64 allocated_len;
	__u64 address;  /* disk address */
	__u64 version;
	__u8 blkbit;
    bool exist;
};

#define SNVME_HELP_GET_NVME_OFFSET	_IOWR('N', 0x1, struct nds_mapping)


#define SNVM_REGISTER_DRIVER	_IO('N', 0x0)
#define SNVM_UNREGISTER_DRIVER	_IO('N', 0x1)

int nds_ext4_retrieve_mapping(struct inode *inode, loff_t offset, loff_t len, struct nds_mapping *mapping);
int nds_retrieve_mapping(struct nds_mapping *mapping);

#endif /* _NVME_H */