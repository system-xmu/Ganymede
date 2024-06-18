# SPDX-License-Identifier: GPL-2.0
obj-m += snvme-core.o
obj-m += snvme.o
# nvme-core.objs := ioctl.o multipath.o core.o
# nvme.objs := pci.o


snvme-core-y				:= core.o ioctl.o
snvme-core-$(CONFIG_NVME_MULTIPATH)	+= multipath.o
snvme-core-$(CONFIG_BLK_DEV_ZONED)	+= zns.o
snvme-core-$(CONFIG_FAULT_INJECTION_DEBUG_FS)	+= fault_inject.o
snvme-core-$(CONFIG_NVME_HWMON)		+= hwmon.o

snvme-objs := nvfs-pci.o pci.o


KERNEL_SRC := /home/qs/linux

# nvme-core-y	:= core.o ioctl.o multipath.o
# nvme-y := pci.o
all:
	$(MAKE) -C $(KERNEL_SRC) M=$(PWD) modules
clean:
	rm -fr *.o *.ko *.mod* .*.cmd .*.d