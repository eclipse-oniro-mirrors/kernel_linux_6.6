// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * nexusland.c - Nexus Shared Memory Land kernel driver
 *
 * Provides a misc device that exposes a randomized, slot-based shared
 * memory region to user-space processes.  Each connection gets a private
 * slot assignment (randomised at alloc time) and a set of zones whose
 * physical pages are mapped into every user VMA that calls mmap().  A
 * background kthread pre-maps zones on demand so that user-space rarely
 * needs to trigger synchronous mappings.
 *
 * Copyright (C) 2025-2026 Huawei Device Co., Ltd.
 */

#include <asm/pgtable.h>
#include <linux/atomic.h>
#include <linux/delay.h>
#include <linux/fs.h>
#include <linux/highmem.h>
#include <linux/init.h>
#include <linux/ioctl.h>
#include <linux/kthread.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/pagemap.h>
#include <linux/printk.h>
#include <linux/random.h>
#include <linux/slab.h>
#include <linux/smp.h>
#include <linux/uaccess.h>

#define DEVICE_NAME "nexusland"

#define IOC_MAGIC 'q'
/* ioctl commands */
#define IOCTL_HELLO      _IOR(IOC_MAGIC,  1, int)
/* Input: conn_key → returns the seed for that connection, or -EPERM */
#define IOCTL_CONN_SEED  _IOWR(IOC_MAGIC, 2, unsigned int)
/* Input: conn_key → synchronously maps ZONE_BATCH zones */
#define IOCTL_ZONE_MAP   _IOR(IOC_MAGIC,  3, unsigned int)
/* Input: conn_key → releases that connection */
#define IOCTL_CONN_FREE  _IOR(IOC_MAGIC,  4, unsigned int)
/* Allocates and returns a new conn_key */
#define IOCTL_CONN_ALLOC _IOW(IOC_MAGIC,  5, unsigned int)
/* Input: conn_key → returns the slot_id for that connection */
#define IOCTL_CONN_BASE  _IOWR(IOC_MAGIC, 6, unsigned int)

/*
 * Layout constants
 *   32 slots, 4 connections
 *   512 subslots, 32 zones
 *   8 zone pages (2 guard pages + 6 content pages)
 *   mmap size = 32 * 512 * 8 * 4k = 512 MiB
 *   actual allocated size = 4 * 32 * 8 * 4k = 4 MiB
 */
#define MAX_SLOT_NUM    16
#define MAX_SUBSLOT_NUM 512
#define MAX_CONN_NUM    4
#define MAX_ZONE_NUM    32
#define ZONE_BATCH      8
/* Number of pages per zone */
#define ZONE_PAGES      8
#define MAX_USERS       64
#define MAX_KEY         1024

enum conn_state {
	CONN_FREE,  /* not yet in use */
	CONN_ALLOC, /* currently allocated */
};

enum zone_state {
	ZONE_UNMAP,     /* not yet mapped into user-space */
	ZONE_FREE,      /* mapped into user-space but not yet claimed */
	ZONE_ALLOC,     /* mapped and in use by user-space */
	ZONE_INTERNAL,  /* transient state between UNMAP and FREE to prevent double-map */
};

/* One 4 KiB page of metadata per connection, shared with user-space */
struct slot_meta {
	int states[MAX_ZONE_NUM];
	char lock;
};

/* Node in the list of active (mmap'd but not yet munmap'd) VMAs */
struct svma_entry {
	struct vm_area_struct *vma;
	struct list_head list;
};

/* Physical pages backing each connection's zones */
static struct page *pages[MAX_CONN_NUM][MAX_ZONE_NUM][ZONE_PAGES];
/* Slot index assigned to each connection */
static unsigned int conns[MAX_CONN_NUM];
/* Subslot index assigned to each zone within a connection */
static unsigned int zones[MAX_CONN_NUM][MAX_ZONE_NUM];

/* Per-connection randomisation seed */
static unsigned int seeds[MAX_CONN_NUM];
/* Per-connection allocation state */
static int conn_states[MAX_CONN_NUM];
/* Pointer to the kernel mapping of each connection's metadata page */
static struct slot_meta *meta[MAX_CONN_NUM];
/* Reverse mapping from conn_key to conn_id; -1 means no mapping */
static int rev_keys[MAX_KEY];

/* Protects conn_states[], meta[], seeds[], conns[], and rev_keys[] */
static DEFINE_MUTEX(conn_lock);
/* Protects svma_list and all remap_pfn_range() calls */
static DEFINE_MUTEX(svma_lock);

/* List of all active (post-mmap, pre-munmap) VMAs */
static LIST_HEAD(svma_list);

/* Background kthread that pre-maps zones asynchronously */
static struct task_struct *nexusland_kthread;
static DEFINE_MUTEX(thread_lock);

/* Simple linear-congruential step used to generate slot/zone indices */
static unsigned int get_next_value(unsigned int previous)
{
	return ((previous * 1103515245U) + 12345U) & 0x7fffffffU;
}

/**
 * conns_init - randomly assign distinct slot indices to each connection
 * @seed: randomisation seed
 *
 * Uses a bitmap to guarantee uniqueness across the MAX_SLOT_NUM slots.
 */
static void conns_init(unsigned int seed)
{
	unsigned int bitmap[MAX_SLOT_NUM / 32 + 1] = {0};
	unsigned int pre_value = seed;
	int i = 0;

	while (i < MAX_CONN_NUM) {
		unsigned int cur_value = get_next_value(pre_value) % MAX_SLOT_NUM;
		int index = cur_value / 32;
		int bit = cur_value % 32;

		if ((bitmap[index] & (1U << bit)) == 0) {
			bitmap[index] |= (1U << bit);
			conns[i++] = cur_value;
		}
		pre_value = cur_value;
	}
}

/**
 * zones_init - randomly assign distinct subslot indices to each zone
 * @conn_id: connection to initialise
 * @seed:    randomisation seed
 *
 * Subslot 0 is reserved (occupied by the metadata page), so the bitmap
 * starts with bit 0 pre-set.
 */
static void zones_init(int conn_id, unsigned int seed)
{
	unsigned int bitmap[MAX_SUBSLOT_NUM / 32 + 1] = {0};
	unsigned int pre_value = seed;
	int i = 1;

	bitmap[0] |= 1U; /* reserve subslot 0 for metadata */

	while (i < MAX_ZONE_NUM) {
		unsigned int cur_value = get_next_value(pre_value) % MAX_SUBSLOT_NUM;
		int index = cur_value / 32;
		int bit = cur_value % 32;

		if ((bitmap[index] & (1U << bit)) == 0) {
			bitmap[index] |= (1U << bit);
			zones[conn_id][i++] = cur_value;
		}
		pre_value = cur_value;
	}
}

static void zones_print(int conn_id)
{
	pr_info("[nexusland] zones[%d]: %u %u %u %u\n",
		conn_id,
		zones[conn_id][1], zones[conn_id][2],
		zones[conn_id][3], zones[conn_id][4]);
}

/**
 * svma_close - vm_ops->close callback; removes the VMA from svma_list
 */
static void svma_close(struct vm_area_struct *vma)
{
	struct svma_entry *entry, *tmp;

	mutex_lock(&svma_lock);
	list_for_each_entry_safe(entry, tmp, &svma_list, list) {
		if (entry->vma == vma) {
			list_del(&entry->list);
			kfree(entry);
			pr_info("[nexusland] munmap (close), vma removed\n");
			break;
		}
	}
	mutex_unlock(&svma_lock);
}

static const struct vm_operations_struct svma_ops = {
	.close = svma_close,
};

/**
 * nexusland_map_zone - map all pages of one zone into a user VMA
 * @vma:     target VMA
 * @conn_id: connection index
 * @zone_id: zone index within the connection
 *
 * Pages are allocated on first use.  The first and last pages of each
 * zone are mapped as guard pages (PAGE_KERNEL / no user access); the
 * interior pages use the VMA's default protection.
 *
 * Must be called with svma_lock held.
 *
 * Returns 0 on success, -EFAULT on allocation or mapping failure.
 */
static int nexusland_map_zone(struct vm_area_struct *vma, int conn_id, int zone_id)
{
	int i;

	for (i = 0; i < ZONE_PAGES; i++) {
		unsigned long pfn, user_va;
		pgprot_t prot;

		if (!pages[conn_id][zone_id][i]) {
			void *page_addr;

			pages[conn_id][zone_id][i] = alloc_page(GFP_KERNEL);
			if (!pages[conn_id][zone_id][i]) {
				pr_info("[nexusland] alloc page (%d, %3d, %d) failed\n",
					conn_id, zone_id, i);
				return -EFAULT;
			}

			/* Zero the page and stamp an identity string for debugging */
			page_addr = kmap_local_page(pages[conn_id][zone_id][i]);
			memset(page_addr, 0, PAGE_SIZE);
			snprintf((char *)page_addr, PAGE_SIZE,
				 "(%d, %3d, %d)", conn_id, zone_id, i);
			kunmap_local(page_addr);
		}

		pfn = page_to_pfn(pages[conn_id][zone_id][i]);
		user_va = vma->vm_start
			+ conns[conn_id] * MAX_SUBSLOT_NUM * ZONE_PAGES * PAGE_SIZE
			+ zones[conn_id][zone_id] * ZONE_PAGES * PAGE_SIZE
			+ i * PAGE_SIZE;

		/* First and last pages act as guard pages; deny user access */
		if (i == 0 || i == ZONE_PAGES - 1)
			prot = PAGE_KERNEL;
		else
			prot = vma->vm_page_prot;

		if (remap_pfn_range(vma, user_va, pfn, PAGE_SIZE, prot)) {
			pr_info("[nexusland] remap page (%d, %3d, %d) failed\n",
				conn_id, zone_id, i);
			return -EFAULT;
		}
	}

	return 0;
}

/**
 * nexusland_zone_map_batch - map up to ZONE_BATCH unmapped zones for a connection
 * @conn_id: connection index
 *
 * Iterates over all zones and maps the first ZONE_BATCH that are in
 * ZONE_UNMAP state, advancing them to ZONE_FREE.
 *
 * Must be called with svma_lock held.
 */
static void nexusland_zone_map_batch(int conn_id)
{
	int *states = meta[conn_id]->states;
	int counter = 0, i;

	for (i = 1; i < MAX_ZONE_NUM; i++) {
		if (states[i] == ZONE_UNMAP) {
			struct svma_entry *entry, *tmp;

			list_for_each_entry_safe(entry, tmp, &svma_list, list)
				nexusland_map_zone(entry->vma, conn_id, i);

			states[i] = ZONE_FREE;
			if (++counter >= ZONE_BATCH)
				break;
		}
	}
}

/**
 * nexusland_map_thread - background kthread body for asynchronous zone mapping
 *
 * Polls all connections and pre-maps zones whenever the number of free
 * zones drops below ZONE_BATCH and there are still unmapped zones left.
 */
static int nexusland_map_thread(void *data)
{
	while (!kthread_should_stop()) {
		int i;

		for (i = 0; i < MAX_CONN_NUM; i++) {
			int free_count = 0, unmap_count = 0, j;
			int *states;

			mutex_lock(&conn_lock);
			if (conn_states[i] != CONN_ALLOC || !meta[i]) {
				mutex_unlock(&conn_lock);
				continue;
			}

			states = meta[i]->states;
			for (j = 1; j < MAX_ZONE_NUM; j++) {
				if (states[j] == ZONE_UNMAP)
					unmap_count++;
				else if (states[j] == ZONE_FREE)
					free_count++;
			}

			if (free_count < ZONE_BATCH && unmap_count > 0) {
				mutex_lock(&svma_lock);
				nexusland_zone_map_batch(i);
				mutex_unlock(&svma_lock);
			}
			mutex_unlock(&conn_lock);
		}
	}

	return 0;
}

/**
 * nexusland_conn_init - initialise a newly allocated connection
 * @conn_id: connection index
 *
 * Allocates the metadata page, maps it (and an initial batch of data
 * zones) into all currently active VMAs.
 *
 * Must be called with svma_lock held.
 *
 * Returns 0 on success, -EFAULT on failure.
 */
static int nexusland_conn_init(int conn_id)
{
	struct svma_entry *entry, *tmp;
	int *states;
	int i;

	/* Allocate and kmap the metadata page */
	pages[conn_id][0][0] = alloc_page(GFP_KERNEL);
	if (!pages[conn_id][0][0]) {
		pr_info("[nexusland] alloc meta page for conn %d failed\n", conn_id);
		return -EFAULT;
	}
	meta[conn_id] = kmap(pages[conn_id][0][0]);

	/* Initialise zone states */
	states = meta[conn_id]->states;
	states[0] = ZONE_ALLOC;
	for (i = 1; i < MAX_ZONE_NUM; i++)
		states[i] = ZONE_UNMAP;

	meta[conn_id]->lock = 0;

	/* Map the metadata page and the initial zone batch into all active VMAs */
	list_for_each_entry_safe(entry, tmp, &svma_list, list) {
		struct vm_area_struct *vma = entry->vma;
		unsigned long pfn = page_to_pfn(pages[conn_id][0][0]);
		unsigned long meta_va = vma->vm_start
			+ conns[conn_id] * MAX_SUBSLOT_NUM * ZONE_PAGES * PAGE_SIZE;

		if (remap_pfn_range(vma, meta_va, pfn, PAGE_SIZE, vma->vm_page_prot)) {
			pr_info("[nexusland] remap meta page for conn %d failed\n", conn_id);
			return -EFAULT;
		}

		for (i = 1; i < ZONE_BATCH; i++) {
			nexusland_map_zone(vma, conn_id, i);
			states[i] = ZONE_FREE;
		}
	}

	return 0;
}

/**
 * nexusland_conn_alloc - allocate a free connection slot
 *
 * Returns the conn_id on success, or -1 if all slots are occupied.
 */
static int nexusland_conn_alloc(void)
{
	int i;

	mutex_lock(&conn_lock);
	for (i = 0; i < MAX_CONN_NUM; i++) {
		if (conn_states[i] == CONN_FREE) {
			conn_states[i] = CONN_ALLOC;
			mutex_unlock(&conn_lock);
			return i;
		}
	}
	mutex_unlock(&conn_lock);
	pr_info("[nexusland] all connections are in use\n");
	return -1;
}

/**
 * nexusland_conn_free - release a connection and free its backing pages
 * @conn_id: connection to release
 */
static void nexusland_conn_free(int conn_id)
{
	int i, j;

	for (i = 0; i < MAX_ZONE_NUM; i++) {
		for (j = 0; j < ZONE_PAGES; j++) {
			if (pages[conn_id][i][j]) {
				__free_page(pages[conn_id][i][j]);
				pages[conn_id][i][j] = NULL;
			}
		}
	}

	meta[conn_id] = NULL;

	mutex_lock(&conn_lock);
	conn_states[conn_id] = CONN_FREE;
	mutex_unlock(&conn_lock);
}

/**
 * nexusland_mmap_init - replay all already-mapped zones into a new VMA
 * @vma:     new user VMA
 * @conn_id: connection whose pages should be replayed
 *
 * Must be called with svma_lock held.
 */
static int nexusland_mmap_init(struct vm_area_struct *vma, int conn_id)
{
	int *states = meta[conn_id]->states;
	int i;

	for (i = 1; i < MAX_ZONE_NUM; i++) {
		if (states[i] == ZONE_FREE || states[i] == ZONE_ALLOC)
			nexusland_map_zone(vma, conn_id, i);
	}

	return 0;
}

/**
 * nexusland_mmap - file_operations->mmap
 *
 * Synchronises the new VMA with the current state of all active
 * connections, registers a vm_ops close handler, and adds the VMA to
 * svma_list.
 */
static int nexusland_mmap(struct file *file, struct vm_area_struct *vma)
{
	struct svma_entry *entry;
	int i;

	/* Replay existing connections into the new VMA */
	for (i = 0; i < MAX_CONN_NUM; i++) {
		if (conn_states[i] == CONN_ALLOC) {
			if (nexusland_mmap_init(vma, i) < 0) {
				pr_info("[nexusland] mmap_init failed for conn %d\n", i);
				return -EFAULT;
			}
		}
	}

	vma->vm_ops = &svma_ops;

	entry = kmalloc(sizeof(*entry), GFP_KERNEL);
	if (!entry)
		return -ENOMEM;

	entry->vma = vma;
	mutex_lock(&svma_lock);
	list_add(&entry->list, &svma_list);
	mutex_unlock(&svma_lock);

	return 0;
}

static long nexusland_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	int ret, conn_id, conn_key;

	switch (cmd) {
	case IOCTL_HELLO:
		if (copy_from_user(&ret, (int __user *)arg, sizeof(ret)))
			return -EFAULT;
		pr_info("[nexusland] hello, %d\n", ret);
		break;

	case IOCTL_CONN_SEED:
		if (copy_from_user(&conn_key, (int __user *)arg, sizeof(conn_key)))
			return -EFAULT;
		if (conn_key < 0 || conn_key >= MAX_KEY)
			return -EINVAL;
		conn_id = rev_keys[conn_key];
		if (conn_id == -1)
			return -EPERM;
		if (copy_to_user((unsigned int __user *)arg,
				 &seeds[conn_id], sizeof(seeds[conn_id])))
			return -EFAULT;
		break;

	case IOCTL_ZONE_MAP:
		if (copy_from_user(&conn_key, (int __user *)arg, sizeof(conn_key)))
			return -EFAULT;
		if (conn_key < 0 || conn_key >= MAX_KEY)
			return -EINVAL;
		conn_id = rev_keys[conn_key];
		if (conn_id == -1)
			return -EPERM;
		mutex_lock(&svma_lock);
		nexusland_zone_map_batch(conn_id);
		mutex_unlock(&svma_lock);
		break;

	case IOCTL_CONN_FREE:
		if (copy_from_user(&conn_key, (int __user *)arg, sizeof(conn_key)))
			return -EFAULT;
		if (conn_key < 0 || conn_key >= MAX_KEY)
			return -EINVAL;
		conn_id = rev_keys[conn_key];
		if (conn_id == -1)
			return -EPERM;
		mutex_lock(&svma_lock);
		nexusland_conn_free(conn_id);
		rev_keys[conn_key] = -1;
		mutex_unlock(&svma_lock);
		break;

	case IOCTL_CONN_ALLOC: {
		unsigned int rnd;

		conn_id = nexusland_conn_alloc();
		if (conn_id < 0)
			return -ENOMEM;

		/* Pick a free key slot using a random starting point */
		mutex_lock(&conn_lock);
		get_random_bytes(&rnd, sizeof(rnd));
		conn_key = (int)(rnd % MAX_KEY);
		while (rev_keys[conn_key] != -1)
			conn_key = (int)(get_next_value((unsigned int)conn_key) % MAX_KEY);
		mutex_unlock(&conn_lock);

		rev_keys[conn_key] = conn_id;

		/* Assign a random seed and initialise zone layout */
		get_random_bytes(&seeds[conn_id], sizeof(seeds[conn_id]));
		zones_init(conn_id, seeds[conn_id]);
		zones_print(conn_id);

		mutex_lock(&svma_lock);
		ret = nexusland_conn_init(conn_id);
		mutex_unlock(&svma_lock);
		if (ret < 0)
			return ret;

		if (copy_to_user((unsigned int __user *)arg,
				 &conn_key, sizeof(conn_key)))
			return -EFAULT;
		break;
	}

	case IOCTL_CONN_BASE:
		if (copy_from_user(&conn_key, (int __user *)arg, sizeof(conn_key)))
			return -EFAULT;
		if (conn_key < 0 || conn_key >= MAX_KEY)
			return -EINVAL;
		conn_id = rev_keys[conn_key];
		if (conn_id == -1)
			return -EPERM;
		if (copy_to_user((unsigned int __user *)arg,
				 &conns[conn_id], sizeof(conns[conn_id])))
			return -EFAULT;
		break;

	default:
		return -ENOTTY;
	}

	return 0;
}

static int nexusland_open(struct inode *inode, struct file *file)
{
	mutex_lock(&thread_lock);
	if (!nexusland_kthread)
		nexusland_kthread = kthread_run(nexusland_map_thread, NULL, "nexusland");
	mutex_unlock(&thread_lock);

	return 0;
}

/**
 * nexusland_release - file_operations->release (called on last close())
 *
 * If no connections remain allocated, stops the background kthread.
 */
static int nexusland_release(struct inode *inode, struct file *file)
{
	int i, ret;

	mutex_lock(&conn_lock);
	for (i = 0; i < MAX_CONN_NUM; i++) {
		if (conn_states[i] == CONN_ALLOC) {
			mutex_unlock(&conn_lock);
			return 0;
		}
	}
	mutex_unlock(&conn_lock);

	/* No connections remain; stop the background thread */
	mutex_lock(&thread_lock);
	if (!nexusland_kthread) {
		pr_info("[nexusland] kthread already stopped\n");
		mutex_unlock(&thread_lock);
		return 0;
	}

	ret = kthread_stop(nexusland_kthread);
	if (ret != -EINTR)
		pr_info("[nexusland] kthread stopped\n");
	else
		pr_info("[nexusland] kthread stop failed\n");

	nexusland_kthread = NULL;
	mutex_unlock(&thread_lock);

	return 0;
}

static const struct file_operations nexusland_fops = {
	.owner          = THIS_MODULE,
	.open           = nexusland_open,
	.release        = nexusland_release,
	.mmap           = nexusland_mmap,
	.unlocked_ioctl = nexusland_ioctl,
};

static struct miscdevice nexusland_dev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name  = DEVICE_NAME,
	.fops  = &nexusland_fops,
};

static int __init nexusland_init(void)
{
	unsigned int seed;
	int i;

	pr_info("[nexusland] initialising\n");

	get_random_bytes(&seed, sizeof(seed));
	conns_init(seed);

	/* No concurrency at this point; no locking needed */
	for (i = 0; i < MAX_CONN_NUM; i++)
		conn_states[i] = CONN_FREE;

	for (i = 0; i < MAX_KEY; i++)
		rev_keys[i] = -1;

	return misc_register(&nexusland_dev);
}

static void __exit nexusland_exit(void)
{
	struct svma_entry *entry, *tmp;
	int i, ret;

	/* Release any connections that user-space forgot to free */
	for (i = 0; i < MAX_CONN_NUM; i++) {
		if (conn_states[i] == CONN_ALLOC)
			nexusland_conn_free(i);
	}

	/* Stop the background kthread if it is still running */
	if (nexusland_kthread) {
		ret = kthread_stop(nexusland_kthread);
		if (ret != -EINTR)
			pr_info("[nexusland] kthread stopped\n");
		else
			pr_info("[nexusland] kthread stop failed\n");
		nexusland_kthread = NULL;
	}

	/* Free the svma list entries */
	list_for_each_entry_safe(entry, tmp, &svma_list, list) {
		list_del(&entry->list);
		kfree(entry);
	}

	pr_info("[nexusland] exiting\n");
	misc_deregister(&nexusland_dev);
}

module_init(nexusland_init);
module_exit(nexusland_exit);

MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Yinpeng Wu <wyp1536481268@foxmail.com>");
MODULE_DESCRIPTION("Nexus Shared Memory Land driver");
