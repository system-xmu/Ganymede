#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

static const struct modversion_info ____versions[]
__used __section("__versions") = {
	{ 0xe74b8d01, "module_layout" },
	{ 0x2f2c95c4, "flush_work" },
	{ 0x3ce4ca6f, "disable_irq" },
	{ 0x5c26a53b, "wait_for_completion_io_timeout" },
	{ 0x289e313c, "dma_map_sg_attrs" },
	{ 0xb47f3031, "cdev_del" },
	{ 0x18e60984, "__do_once_start" },
	{ 0x7409f913, "kmalloc_caches" },
	{ 0x27f3788c, "__nvme_check_ready" },
	{ 0xeb233a45, "__kmalloc" },
	{ 0x7ae57c54, "cdev_init" },
	{ 0xf888ca21, "sg_init_table" },
	{ 0x1bc872d6, "put_devmap_managed_page" },
	{ 0xc96869c9, "nvme_get_features" },
	{ 0x4500ed09, "blk_cleanup_queue" },
	{ 0x1c858bc9, "pci_free_irq_vectors" },
	{ 0xb9e6f44b, "nvme_reset_ctrl" },
	{ 0xc596fd0f, "nvme_init_ctrl_finish" },
	{ 0xe7d38de5, "nvme_put_ns" },
	{ 0x7aa1756e, "kvfree" },
	{ 0x4c5138af, "param_ops_int" },
	{ 0xf399814e, "pci_sriov_configure_simple" },
	{ 0x42e2f88c, "device_release_driver" },
	{ 0x46cf10eb, "cachemode2protval" },
	{ 0x1a7e91bf, "dma_unmap_sg_attrs" },
	{ 0xe0fb6ca, "nvme_wait_reset" },
	{ 0x39d3e2cd, "dma_set_mask" },
	{ 0xa92ec74, "boot_cpu_data" },
	{ 0xca647146, "nvme_stop_ctrl" },
	{ 0xf9b94938, "pci_disable_device" },
	{ 0x2190ff42, "nvme_unfreeze" },
	{ 0xd6132edd, "blk_mq_tagset_busy_iter" },
	{ 0xe2176f9e, "blk_mq_start_request" },
	{ 0x18859b6e, "nvme_set_features" },
	{ 0x4d4d7b79, "blk_mq_map_queues" },
	{ 0xc81ff366, "nvme_shutdown_ctrl" },
	{ 0xd90d61a, "seq_printf" },
	{ 0x56470118, "__warn_printk" },
	{ 0xaa688b6, "nvme_set_queue_count" },
	{ 0xde84ed02, "device_destroy" },
	{ 0x996e157f, "pci_get_class" },
	{ 0x9034a696, "mempool_destroy" },
	{ 0x3a739890, "nvme_stop_queues" },
	{ 0x87b8798d, "sg_next" },
	{ 0xa8196862, "blk_mq_tag_to_rq" },
	{ 0x265ab779, "nvme_complete_async_event" },
	{ 0x2c9a3ae8, "param_ops_bool" },
	{ 0x3213f038, "mutex_unlock" },
	{ 0x6091b333, "unregister_chrdev_region" },
	{ 0xc8465159, "nvme_kill_queues" },
	{ 0x784db36, "dma_free_attrs" },
	{ 0x92c83a0, "param_set_uint_minmax" },
	{ 0x78ddb76b, "dmi_match" },
	{ 0xb5aa7165, "dma_pool_destroy" },
	{ 0x97651e6c, "vmemmap_base" },
	{ 0x26f4100b, "blk_mq_update_nr_hw_queues" },
	{ 0xcde77bcc, "free_opal_dev" },
	{ 0x176abeea, "sysfs_remove_group" },
	{ 0x297038e0, "pv_ops" },
	{ 0x42635d55, "pm_suspend_global_flags" },
	{ 0xffc2e973, "dma_set_coherent_mask" },
	{ 0xcbfb33e4, "init_opal_dev" },
	{ 0xbb9ed3bf, "mutex_trylock" },
	{ 0x8a9c70ed, "nvme_sec_submit" },
	{ 0x69d43b5b, "blk_mq_init_queue" },
	{ 0x6b10bee1, "_copy_to_user" },
	{ 0x17de3d5, "nr_cpu_ids" },
	{ 0x5295080c, "pci_set_master" },
	{ 0x130850d2, "pci_alloc_irq_vectors_affinity" },
	{ 0xc81b2e71, "_dev_warn" },
	{ 0x2a253fb0, "pci_enable_pcie_error_reporting" },
	{ 0xdf0aba5b, "nvme_try_sched_reset" },
	{ 0x9e683f75, "__cpu_possible_mask" },
	{ 0xf5a42764, "nvme_enable_ctrl" },
	{ 0x124bad4d, "kstrtobool" },
	{ 0x126e0160, "pci_restore_state" },
	{ 0x300fef5d, "blk_put_queue" },
	{ 0x5460bc6d, "current_task" },
	{ 0x813cf212, "nvme_io_timeout" },
	{ 0xcefb0c9f, "__mutex_init" },
	{ 0x51641162, "opal_unlock_from_suspend" },
	{ 0x4ee5b8be, "sysfs_create_group" },
	{ 0x8213b7af, "nvme_find_get_ns" },
	{ 0x22587461, "blk_mq_free_tag_set" },
	{ 0xde80cd09, "ioremap" },
	{ 0x69994047, "nvme_remove_namespaces" },
	{ 0xe93dc72, "__blk_rq_map_sg" },
	{ 0xe9132392, "acpi_storage_d3" },
	{ 0x517f3e99, "pci_read_config_word" },
	{ 0xdb7fc42e, "dma_alloc_attrs" },
	{ 0x22b75da0, "pci_device_is_present" },
	{ 0x4dfa8d4b, "mutex_lock" },
	{ 0x79086292, "pci_get_domain_bus_and_slot" },
	{ 0xaa3d85ff, "device_create" },
	{ 0x2f7754a8, "dma_pool_free" },
	{ 0x5b9bcaea, "blk_execute_rq_nowait" },
	{ 0x68911183, "pci_load_saved_state" },
	{ 0xcc328cf1, "pci_request_irq" },
	{ 0xaa82faa0, "_dev_err" },
	{ 0x42160169, "flush_workqueue" },
	{ 0x868784cb, "__symbol_get" },
	{ 0xe4ae1693, "nvme_fail_nonready_command" },
	{ 0x46c47fb6, "__node_distance" },
	{ 0xf0dc05df, "cdev_add" },
	{ 0x599fb41c, "kvmalloc_node" },
	{ 0x78ac0e7c, "blk_get_queue" },
	{ 0x1371007e, "nvme_init_ctrl" },
	{ 0xc3b4c22d, "pci_select_bars" },
	{ 0x341c239b, "_dev_info" },
	{ 0x4066b375, "kmem_cache_alloc_node_trace" },
	{ 0x91789453, "nvme_change_ctrl_state" },
	{ 0xc6a5f117, "blk_mq_free_request" },
	{ 0xc3762aec, "mempool_alloc" },
	{ 0x9c122bcf, "mempool_create_node" },
	{ 0xc21eca5c, "pci_free_irq" },
	{ 0x708528a6, "put_device" },
	{ 0x47f24be1, "dma_max_mapping_size" },
	{ 0xd0da656b, "__stack_chk_fail" },
	{ 0x23c29d00, "get_user_pages" },
	{ 0xb4089956, "nvme_sync_queues" },
	{ 0x6d0bb898, "param_get_uint" },
	{ 0x4ab8d871, "nvme_cleanup_cmd" },
	{ 0x1d24c881, "___ratelimit" },
	{ 0x92997ed8, "_printk" },
	{ 0x4dfd131a, "nvme_cancel_request" },
	{ 0xbb07cf23, "nvme_wait_freeze" },
	{ 0x1343d622, "blk_mq_pci_map_queues" },
	{ 0xef8bf981, "dma_map_page_attrs" },
	{ 0xf5f370e0, "async_schedule_node" },
	{ 0x2ea2c95c, "__x86_indirect_thunk_rax" },
	{ 0x6a037cf1, "mempool_kfree" },
	{ 0x678b96ec, "dma_pool_alloc" },
	{ 0x66b3c7c9, "blk_mq_alloc_tag_set" },
	{ 0xe783e261, "sysfs_emit" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0x8616736b, "pci_unregister_driver" },
	{ 0xc1cf7826, "kmem_cache_alloc_trace" },
	{ 0xa897e3e7, "mempool_free" },
	{ 0x9493fc86, "node_states" },
	{ 0xba8fbd64, "_raw_spin_lock" },
	{ 0x5ccef533, "nvme_disable_ctrl" },
	{ 0xb5222b73, "get_device" },
	{ 0x53ed3f9d, "pci_irq_vector" },
	{ 0x64b62862, "nvme_wq" },
	{ 0xd35a6d31, "mempool_kmalloc" },
	{ 0xe62f4dea, "pci_disable_pcie_error_reporting" },
	{ 0xfcec0987, "enable_irq" },
	{ 0x37a0cba, "kfree" },
	{ 0x3b6c41ea, "kstrtouint" },
	{ 0x47110967, "blk_mq_quiesce_queue" },
	{ 0x51e17ed6, "nvme_submit_sync_cmd" },
	{ 0xeeaca2b9, "param_set_uint" },
	{ 0x92f123db, "blk_mq_unquiesce_queue" },
	{ 0xedc03953, "iounmap" },
	{ 0x4f9fe81c, "pcibios_resource_to_bus" },
	{ 0xb09e796, "nvme_start_ctrl" },
	{ 0x38fb7e10, "__pci_register_driver" },
	{ 0xd3d379df, "nvme_start_freeze" },
	{ 0x51641f24, "nvme_setup_cmd" },
	{ 0x9f91b47b, "class_destroy" },
	{ 0xae927477, "dma_unmap_page_attrs" },
	{ 0xd844d18a, "blk_mq_complete_request_remote" },
	{ 0xd45434ee, "admin_timeout" },
	{ 0x9a353ae, "__x86_indirect_alt_call_rax" },
	{ 0x502c52af, "vm_iomap_memory" },
	{ 0x608741b5, "__init_swait_queue_head" },
	{ 0x63c4d61f, "__bitmap_weight" },
	{ 0xd851e829, "nvme_alloc_request" },
	{ 0xa227aa93, "nvme_start_queues" },
	{ 0x8810754a, "_find_first_bit" },
	{ 0xb841a026, "blk_mq_tagset_wait_completed_request" },
	{ 0x1ba59527, "__kmalloc_node" },
	{ 0xa6dad321, "pci_dev_put" },
	{ 0x6e9dd606, "__symbol_put" },
	{ 0xc5b6f236, "queue_work_on" },
	{ 0xa6257a2f, "complete" },
	{ 0x656e4a6e, "snprintf" },
	{ 0xb07c6660, "pci_enable_device_mem" },
	{ 0xdb9e75aa, "nvme_wait_freeze_timeout" },
	{ 0x9be59030, "pci_release_selected_regions" },
	{ 0xe27987f8, "pci_request_selected_regions" },
	{ 0x13c49cc2, "_copy_from_user" },
	{ 0x1efddd61, "nvme_complete_rq" },
	{ 0x83256e75, "dma_pool_create" },
	{ 0x58e8dcc8, "param_ops_uint" },
	{ 0x2aa37818, "__class_create" },
	{ 0xe98553d7, "nvme_uninit_ctrl" },
	{ 0xcf8b5edd, "pci_find_ext_capability" },
	{ 0x88db9f48, "__check_object_size" },
	{ 0xe3ec2f2b, "alloc_chrdev_region" },
	{ 0xec2181a2, "__put_page" },
	{ 0x16825a8d, "pcie_capability_read_word" },
	{ 0x5094de0c, "pcie_aspm_enabled" },
	{ 0x8b83033c, "pci_save_state" },
	{ 0x3674b57b, "__do_once_done" },
	{ 0x587f22d7, "devmap_managed_key" },
};

MODULE_INFO(depends, "snvme-core");

MODULE_ALIAS("pci:v00008086d00000953sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00008086d00000A53sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00008086d00000A54sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00008086d00000A55sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00008086d0000F1A5sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00008086d0000F1A6sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00008086d00005845sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v0000126Fd00002263sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00001BB1d00000100sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00001C58d00000003sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00001C58d00000023sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00001C5Fd00000540sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v0000144Dd0000A821sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v0000144Dd0000A822sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00001987d00005016sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00001B4Bd00001092sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000010ECd00005762sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00001CC1d00008201sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00001C5Cd00001504sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v000015B7d00002001sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00001D97d00002263sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00002646d00002262sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00002646d00002263sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00001D0Fd00000061sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00001D0Fd00000065sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00001D0Fd00008061sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00001D0Fd0000CD00sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00001D0Fd0000CD01sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v00001D0Fd0000CD02sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v0000106Bd00002001sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v0000106Bd00002003sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v0000106Bd00002005sv*sd*bc*sc*i*");
MODULE_ALIAS("pci:v*d*sv*sd*bc01sc08i02*");

MODULE_INFO(srcversion, "E7C3EE1AF9BB62867CDA016");
