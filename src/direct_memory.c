// SPDX-License-Identifier: GPL-3.0-or-later
// Where the app's long-lived memory goes in the pool the CPU shares with the GPU.
//
// The system gives out direct memory from the bottom of the pool: the lowest room that fits.
// A game's own memory (4 GiB) and its graphics come and go with each session, while two kinds
// of blocks stay for good: the pieces the heap grows by (headless/heap_arenas.inc) and the
// stacks of threads (the Payload SDK takes 2 MiB of direct memory for each, and keeps freed
// ones for the next threads). Taken from the bottom like the rest, those landed between a
// running game's graphics memory, and when the game ended they were left spread over the
// middle of the pool. The next game's graphics get the largest free stretch, which was then
// much smaller than what was free in total, and smaller again after every game.
//
// So blocks that stay are taken from the top of the pool downwards, and everything a session
// takes and gives back is left to the system as before. The link wraps the calls
// (tools/link-headless-native.sh): the heap asks for the top itself (eden_direct_high), and
// the other blocks that stay are known by their size: a thread's stack is 2 MiB, and so is the
// one block the graphics driver keeps for good (its zero block, taken in the middle of a game
// and then left where it was). The driver's command ring and marker page are known by who
// asks. The app's own 2 MiB blocks, the slots of the guest page tables, go back when a game
// ends: they bypass the wrap; EDEN_GUEST_BAND2 optionally prefers a fixed bottom band.
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int __real_sceKernelAllocateDirectMemory(int64_t, int64_t, size_t, size_t, int, int64_t *);
int __real_sceKernelReleaseDirectMemory(int64_t, size_t);
int64_t sceKernelGetDirectMemorySize(void);
int32_t sceKernelAvailableDirectMemorySize(int64_t, int64_t, size_t, int64_t *, size_t *);
int sceKernelUsleep(unsigned int microseconds);

#define EDEN_DIRECT_STEP ((int64_t)2 << 20)       /* blocks at the top are placed in 2 MiB steps */
#define EDEN_DIRECT_SMALL ((size_t)2 << 20)       /* a thread's stack; the driver's zero block */

/* What is out of the pool, by size: under 2 MiB, 2 MiB, up to 32 MiB, more. Every request of the
 * program and the driver passes here, so the reports can say what the owners they name leave out
 * (at a refusal in run 53 they named 10.8 GiB of 12.1). */
static atomic_ullong eden_direct_bytes[4], eden_direct_blocks[4];
static atomic_ullong eden_band_hits, eden_band_fallbacks;

static unsigned eden_direct_class(size_t size) {
  return size < EDEN_DIRECT_SMALL ? 0 : size == EDEN_DIRECT_SMALL ? 1 : size <= ((size_t)32 << 20) ? 2 : 3;
}

/* ---- The pool governor's ledger (OOT-PS5 memory design, phase 0) ----
 * The same requests by owner: the game's memory, Eden's heap, the graphics driver, the JIT's code,
 * the guest page tables, thread stacks and the rest. Every crash of Ocarina of Time on the console
 * so far ended with the pool's largest free block at 0-4 MiB, each time in a different owner (a page
 * table commit, a staging buffer, the driver, the JIT): whoever asked last. The owners' own reports
 * count differently (the driver counts its memory twice); this counts what each one took out of the
 * pool. An owner is told by the caller, or for the wrapped system call by who asks: the driver's
 * allocator, a 2 MiB block (a thread's stack), the rest. A block is recorded by its start; a release
 * from inside a block (none is known) is counted as untracked. No thread-local state: the first use
 * of emulated TLS allocates, and these calls come from inside the heap's growth too. */
enum {
  EDEN_OWNER_OTHER, EDEN_OWNER_GUEST, EDEN_OWNER_HEAP, EDEN_OWNER_DRIVER, EDEN_OWNER_JIT,
  EDEN_OWNER_TABLES, EDEN_OWNER_STACK, EDEN_OWNERS
};
static const char *const eden_owner_names[EDEN_OWNERS] = {"other", "guest", "heap", "driver", "jit", "tables", "stacks"};
static atomic_ullong eden_owner_bytes[EDEN_OWNERS];
static atomic_ullong eden_ledger_untracked; /* bytes given back that no record held */

#define EDEN_LEDGER_SLOTS 32768u /* power of two; ~6100 blocks of 2 MiB fill the 12 GiB pool */
static int64_t eden_ledger_start[EDEN_LEDGER_SLOTS];
static uint64_t eden_ledger_size[EDEN_LEDGER_SLOTS];
static uint8_t eden_ledger_owner[EDEN_LEDGER_SLOTS];
static uint8_t eden_ledger_used[EDEN_LEDGER_SLOTS]; /* 0 empty, 1 a block, 2 a block that went */
static unsigned eden_ledger_count, eden_ledger_dead;
static atomic_flag eden_ledger_lock = ATOMIC_FLAG_INIT;

static unsigned eden_ledger_hash(int64_t start) {
  return (unsigned)(((uint64_t)start >> 14) * 0x9E3779B97F4A7C15ull >> 49) & (EDEN_LEDGER_SLOTS - 1);
}
static void eden_ledger_acquire(void) {
  while (atomic_flag_test_and_set_explicit(&eden_ledger_lock, memory_order_acquire)) {}
}
static void eden_ledger_release(void) { atomic_flag_clear_explicit(&eden_ledger_lock, memory_order_release); }

/* Under eden_ledger_lock. Starts over when tombstones fill the table (records are kept). */
static void eden_ledger_insert(int64_t start, uint64_t size, unsigned owner) {
  if (eden_ledger_count + eden_ledger_dead >= EDEN_LEDGER_SLOTS / 4 * 3) {
    if (eden_ledger_count >= EDEN_LEDGER_SLOTS / 2) return; /* full: this block goes untracked */
    for (unsigned i = 0; i < EDEN_LEDGER_SLOTS; ++i) if (eden_ledger_used[i] == 2) eden_ledger_used[i] = 0;
    eden_ledger_dead = 0;
  }
  for (unsigned i = eden_ledger_hash(start), n = 0; n < EDEN_LEDGER_SLOTS; ++n, i = (i + 1) & (EDEN_LEDGER_SLOTS - 1)) {
    if (eden_ledger_used[i] == 1) continue;
    if (eden_ledger_used[i] == 2) --eden_ledger_dead;
    eden_ledger_used[i] = 1;
    eden_ledger_start[i] = start;
    eden_ledger_size[i] = size;
    eden_ledger_owner[i] = (uint8_t)owner;
    ++eden_ledger_count;
    return;
  }
}
/* Under eden_ledger_lock: the record that starts at `start`, or EDEN_LEDGER_SLOTS. */
static unsigned eden_ledger_find(int64_t start) {
  for (unsigned i = eden_ledger_hash(start), n = 0; n < EDEN_LEDGER_SLOTS; ++n, i = (i + 1) & (EDEN_LEDGER_SLOTS - 1)) {
    if (eden_ledger_used[i] == 0) return EDEN_LEDGER_SLOTS;
    if (eden_ledger_used[i] == 1 && eden_ledger_start[i] == start) return i;
  }
  return EDEN_LEDGER_SLOTS;
}
static void eden_ledger_took(int64_t start, size_t size, unsigned owner) {
  atomic_fetch_add_explicit(&eden_owner_bytes[owner], size, memory_order_relaxed);
  eden_ledger_acquire();
  eden_ledger_insert(start, size, owner);
  eden_ledger_release();
}
/* A release of [start, start + size): the front of a recorded block (all of it, or its first part). */
static void eden_ledger_gave(int64_t start, size_t size) {
  eden_ledger_acquire();
  const unsigned i = eden_ledger_find(start);
  if (i == EDEN_LEDGER_SLOTS) {
    eden_ledger_release();
    atomic_fetch_add_explicit(&eden_ledger_untracked, size, memory_order_relaxed);
    return;
  }
  const unsigned owner = eden_ledger_owner[i];
  const uint64_t recorded = eden_ledger_size[i];
  const uint64_t gone = size < recorded ? size : recorded;
  eden_ledger_used[i] = 2;
  --eden_ledger_count;
  ++eden_ledger_dead;
  if (gone < recorded) eden_ledger_insert(start + (int64_t)gone, recorded - gone, owner);
  eden_ledger_release();
  atomic_fetch_sub_explicit(&eden_owner_bytes[owner], gone, memory_order_relaxed);
  if (size > gone) atomic_fetch_add_explicit(&eden_ledger_untracked, size - gone, memory_order_relaxed);
}

static int eden_direct_take_for(unsigned owner, int64_t from, int64_t to, size_t size, size_t alignment, int type,
                                int64_t *start) {
  const int result = __real_sceKernelAllocateDirectMemory(from, to, size, alignment, type, start);
  if (result == 0) {
    atomic_fetch_add_explicit(&eden_direct_bytes[eden_direct_class(size)], size, memory_order_relaxed);
    atomic_fetch_add_explicit(&eden_direct_blocks[eden_direct_class(size)], 1, memory_order_relaxed);
    if (start) eden_ledger_took(*start, size, owner < EDEN_OWNERS ? owner : EDEN_OWNER_OTHER);
  }
  return result;
}
void eden_direct_ledger(unsigned long long *bytes, unsigned long long *blocks,
                        unsigned long long *hits, unsigned long long *fallbacks) {
  *hits = atomic_load_explicit(&eden_band_hits, memory_order_relaxed);
  *fallbacks = atomic_load_explicit(&eden_band_fallbacks, memory_order_relaxed);
  for (unsigned i = 0; i < 4; ++i) {
    bytes[i] = atomic_load_explicit(&eden_direct_bytes[i], memory_order_relaxed);
    blocks[i] = atomic_load_explicit(&eden_direct_blocks[i], memory_order_relaxed);
  }
}

static atomic_flag eden_direct_lock = ATOMIC_FLAG_INIT;
static int64_t eden_direct_floor;                 /* the lowest block taken at the top; 0: none yet */
static int64_t eden_direct_holes[1024];           /* 2 MiB blocks given back above the floor */
static unsigned eden_direct_hole_count;

static void eden_direct_acquire(void) {
  while (atomic_flag_test_and_set_explicit(&eden_direct_lock, memory_order_acquire)) {}
}
static void eden_direct_release(void) { atomic_flag_clear_explicit(&eden_direct_lock, memory_order_release); }

/* Called under eden_direct_lock. Configuration is fixed before any placement preference.
 * Round down to whole guest slots; reject a band that leaves no outside slot. */
static int64_t eden_direct_band2(int64_t total) {
  static int64_t band = -1;
  if (band < 0) {
    const char *value = getenv("EDEN_GUEST_BAND2");
    char *end = NULL;
    const unsigned long long mib = value && value[0] >= '0' && value[0] <= '9'
                                    ? strtoull(value, &end, 10) : 0;
    band = end && *end == '\0' && total > EDEN_DIRECT_STEP &&
                   mib <= (unsigned long long)((total - EDEN_DIRECT_STEP) >> 20)
             ? (int64_t)(mib / 2 * 2 << 20) : 0;
  }
  return band;
}

int eden_direct_low(size_t size, size_t alignment, int type, int64_t *start);
static int eden_direct_low_for(unsigned owner, size_t size, size_t alignment, int type, int64_t *start);

/* Memory from as high in the pool as there is room; from anywhere when the top has none. */
static int eden_direct_high_for(unsigned owner, size_t size, size_t alignment, int type, int64_t *start) {
  const int64_t total = sceKernelGetDirectMemorySize();
  const int64_t bytes = (int64_t)size;
  if (bytes <= 0 || total <= 0) return eden_direct_take_for(owner, 0, total, size, alignment, type, start);
  eden_direct_acquire();
  const int64_t band = eden_direct_band2(total);
  /* A block given back earlier left a hole of its size. */
  while (size == EDEN_DIRECT_SMALL && eden_direct_hole_count != 0) {
    const int64_t hole = eden_direct_holes[--eden_direct_hole_count];
    if (hole >= band && eden_direct_take_for(owner, hole, hole + bytes, size, alignment, type, start) == 0) {
      eden_direct_release();
      return 0;
    }
  }
  /* Below what is at the top already; past what something else has put there, for a while. */
  int64_t next = eden_direct_floor != 0 ? eden_direct_floor : total;
  for (unsigned tries = 0; tries < 512 && next >= bytes && next - bytes >= band; ++tries) {
    const int64_t at = (next - bytes) / EDEN_DIRECT_STEP * EDEN_DIRECT_STEP;
    next = at;
    if (eden_direct_take_for(owner, at, at + bytes, size, alignment, type, start) == 0) {
      eden_direct_floor = at;
      eden_direct_release();
      return 0;
    }
  }
  eden_direct_release();
  return eden_direct_low_for(owner, size, alignment, type, start);
}
/* The heap's pieces (memory_pages.cpp) and the other callers that name no owner. */
int eden_direct_high(size_t size, size_t alignment, int type, int64_t *start) {
  return eden_direct_high_for(EDEN_OWNER_HEAP, size, alignment, type, start);
}

/* Session/graphics allocations prefer outside the guest band; full-pool recovery stays. */
static int eden_direct_low_for(unsigned owner, size_t size, size_t alignment, int type, int64_t *start) {
  const int64_t total = sceKernelGetDirectMemorySize();
  eden_direct_acquire();
  const int64_t band = eden_direct_band2(total);
  eden_direct_release();
  if (band > 0 && eden_direct_take_for(owner, band, total, size, alignment, type, start) == 0) return 0;
  return eden_direct_take_for(owner, 0, total, size, alignment, type, start);
}
int eden_direct_low(size_t size, size_t alignment, int type, int64_t *start) {
  return eden_direct_low_for(EDEN_OWNER_OTHER, size, alignment, type, start);
}
/* The same with the owner named (memory_pages.cpp: the JIT's code, the guest page tables, the heap). */
int eden_direct_owned(int owner, int high, size_t size, size_t alignment, int type, int64_t *start) {
  const unsigned who = owner >= 0 && owner < EDEN_OWNERS ? (unsigned)owner : EDEN_OWNER_OTHER;
  return high ? eden_direct_high_for(who, size, alignment, type, start)
              : eden_direct_low_for(who, size, alignment, type, start);
}

/* Guest slots and packed blocks prefer [0, band). No placeholder or shared high floor.
 * Both owners may cross the boundary on failure: this is deliberately a soft partition. */
int eden_direct_guest(size_t size, size_t alignment, int type, int64_t *start) {
  const int64_t total = sceKernelGetDirectMemorySize();
  eden_direct_acquire();
  const int64_t band = eden_direct_band2(total);
  if (band > 0) {
    if (eden_direct_take_for(EDEN_OWNER_GUEST, 0, band, size, alignment, type, start) == 0) {
      atomic_fetch_add_explicit(&eden_band_hits, 1, memory_order_relaxed);
      eden_direct_release();
      return 0;
    }
    atomic_fetch_add_explicit(&eden_band_fallbacks, 1, memory_order_relaxed);
  }
  eden_direct_release();
  return eden_direct_take_for(EDEN_OWNER_GUEST, 0, total, size, alignment, type, start);
}

/* The Vulkan driver makes its command ring (16 MiB) and, right after it, the ring's marker page
 * once, with the first device, and keeps them until the app ends. Taken from the bottom like a
 * session's memory they landed above the first game's (the ring at 4.6 GiB), and every later
 * game found the pool's largest free stretch cut from 10.3 GiB to 5.7. They are known by who
 * asks (the driver's allocator; no such symbol in a build without that driver), by their sizes
 * and by being the first of those sizes in that order. */
extern void eden_radv_private_radv_ps5_memory_alloc(void) __attribute__((weak));
#define EDEN_DIRECT_RING ((size_t)16 << 20)
#define EDEN_DIRECT_PAGE ((size_t)16 << 10)
static atomic_int eden_direct_driver_kept; /* 0: neither yet; 1: the ring; 2: the marker page too */

static int eden_direct_driver_keeps(size_t size, const void *caller) {
  const char *const allocator = (const char *)(uintptr_t)&eden_radv_private_radv_ps5_memory_alloc;
  if (!allocator || (const char *)caller < allocator || (const char *)caller >= allocator + 2048) return 0;
  int kept = atomic_load_explicit(&eden_direct_driver_kept, memory_order_relaxed);
  if (kept == 0 && size == EDEN_DIRECT_RING)
    return atomic_compare_exchange_strong(&eden_direct_driver_kept, &kept, 1);
  if (kept == 1 && size == EDEN_DIRECT_PAGE)
    return atomic_compare_exchange_strong(&eden_direct_driver_kept, &kept, 2);
  return 0;
}

/* The ledger's owner of a wrapped request: the driver's allocator, else a 2 MiB block (in practice
 * the SDK's thread stacks), else the rest. Driver allocations made outside that function count as
 * "other". */
static unsigned eden_direct_wrapped_owner(size_t size, const void *caller) {
  const char *const allocator = (const char *)(uintptr_t)&eden_radv_private_radv_ps5_memory_alloc;
  if (allocator && (const char *)caller >= allocator && (const char *)caller < allocator + 2048) return EDEN_OWNER_DRIVER;
  return size == EDEN_DIRECT_SMALL ? EDEN_OWNER_STACK : EDEN_OWNER_OTHER;
}

int __wrap_sceKernelAllocateDirectMemory(int64_t from, int64_t to, size_t size, size_t alignment, int type,
                                         int64_t *start) {
  const void *const caller = __builtin_return_address(0);
  const unsigned owner = eden_direct_wrapped_owner(size, caller);
  /* A thread's stack or the driver's zero block, asked for anywhere in the pool; and the
   * driver's command ring and marker page. */
  if (from == 0 && start && to >= sceKernelGetDirectMemorySize() &&
      (size == EDEN_DIRECT_SMALL || eden_direct_driver_keeps(size, caller)))
    return eden_direct_high_for(owner, size, alignment, type, start);
  if (from == 0 && start && to >= sceKernelGetDirectMemorySize())
    return eden_direct_low_for(owner, size, alignment, type, start);
  return eden_direct_take_for(owner, from, to, size, alignment, type, start);
}

int __wrap_sceKernelReleaseDirectMemory(int64_t start, size_t size) {
  const int result = __real_sceKernelReleaseDirectMemory(start, size);
  if (result != 0) return result;
  eden_ledger_gave(start, size);
  atomic_fetch_sub_explicit(&eden_direct_bytes[eden_direct_class(size)], size, memory_order_relaxed);
  atomic_fetch_sub_explicit(&eden_direct_blocks[eden_direct_class(size)], 1, memory_order_relaxed);
  eden_direct_acquire();
  if (size == EDEN_DIRECT_SMALL && eden_direct_floor != 0 && start >= eden_direct_floor &&
      eden_direct_hole_count < sizeof(eden_direct_holes) / sizeof(eden_direct_holes[0]))
    eden_direct_holes[eden_direct_hole_count++] = start;
  eden_direct_release();
  return result;
}

/* ---- The pool governor: levels, the wait for memory, reclaimers (OOT-PS5 memory design, phase 2) ----
 * The pool's largest free block, looked at again every 50 ms at most, gives a level: 0 green,
 * 1 yellow (under EDEN_GOV_YELLOW_MIB, default 1536), 2 red (under EDEN_GOV_RED_MIB, default 512),
 * 3 black (someone waits for memory now). A request the pool refused, where giving up would end the
 * session (a guest page table's slot, a buffer), waits instead: eden_governor_wait asks every
 * registered reclaimer to give memory back (the heap's empty spaces, the guest's reserve above its
 * floor, cold guest memory), every 100 ms, until the pool has a block of the size asked for or the
 * time is up. While someone waits, the caches see the pool short and evict at their hardest
 * (headless/performance.cpp, GuestMemoryWanted). */
static uint64_t eden_gov_now_ns(void) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}
static size_t eden_gov_env_mib(const char *name, size_t fallback) {
  const char *const value = getenv(name);
  if (!value || value[0] < '0' || value[0] > '9') return fallback;
  return (size_t)strtoull(value, NULL, 10);
}
static atomic_ullong eden_gov_largest, eden_gov_checked_ns;
static atomic_uint eden_gov_waiters;
static atomic_ullong eden_gov_waits, eden_gov_won, eden_gov_lost, eden_gov_wait_ms, eden_gov_reclaimed;
static atomic_ullong eden_gov_waits_by[EDEN_OWNERS];

/* The pool's largest free block: looked at now with `fresh`, else at most 50 ms old. 0: not known. */
size_t eden_governor_largest_free(int fresh) {
  const uint64_t now = eden_gov_now_ns();
  const uint64_t checked = atomic_load_explicit(&eden_gov_checked_ns, memory_order_relaxed);
  if (!fresh && checked != 0 && now - checked < 50000000ull)
    return (size_t)atomic_load_explicit(&eden_gov_largest, memory_order_relaxed);
  int64_t first = 0;
  size_t largest = 0;
  const int64_t total = sceKernelGetDirectMemorySize();
  if (total <= 0 || sceKernelAvailableDirectMemorySize(0, total, 0x4000, &first, &largest) != 0) return 0;
  atomic_store_explicit(&eden_gov_largest, largest, memory_order_relaxed);
  atomic_store_explicit(&eden_gov_checked_ns, now, memory_order_relaxed);
  return largest;
}

int eden_governor_level(void) {
  static size_t yellow = 0, red = 0;
  if (yellow == 0) {
    red = eden_gov_env_mib("EDEN_GOV_RED_MIB", 512) << 20;
    yellow = eden_gov_env_mib("EDEN_GOV_YELLOW_MIB", 1536) << 20;
  }
  if (atomic_load_explicit(&eden_gov_waiters, memory_order_relaxed) != 0) return 3;
  const size_t largest = eden_governor_largest_free(0);
  if (largest == 0) return 0; /* not known */
  return largest < red ? 2 : largest < yellow ? 1 : 0;
}

/* Requests waiting for memory now: the caches give way while there are any. */
unsigned eden_governor_wanted(void) { return atomic_load_explicit(&eden_gov_waiters, memory_order_relaxed); }

typedef size_t (*eden_governor_reclaimer)(size_t want);
#define EDEN_GOV_RECLAIMERS 8
static _Atomic(eden_governor_reclaimer) eden_gov_reclaimers[EDEN_GOV_RECLAIMERS];
static const char *eden_gov_reclaimer_names[EDEN_GOV_RECLAIMERS];
static atomic_ullong eden_gov_reclaimer_bytes[EDEN_GOV_RECLAIMERS];
static atomic_uint eden_gov_reclaimer_count;

/* Called once per reclaimer, in the order they should be asked (cheapest first). */
void eden_governor_register(const char *name, eden_governor_reclaimer reclaimer) {
  const unsigned index = atomic_fetch_add_explicit(&eden_gov_reclaimer_count, 1, memory_order_relaxed);
  if (index >= EDEN_GOV_RECLAIMERS) return;
  eden_gov_reclaimer_names[index] = name;
  atomic_store_explicit(&eden_gov_reclaimers[index], reclaimer, memory_order_release);
}

/* Asks the reclaimers in order until `want` bytes came back. What came back. */
static size_t eden_governor_reclaim(size_t want) {
  size_t freed = 0;
  unsigned count = atomic_load_explicit(&eden_gov_reclaimer_count, memory_order_relaxed);
  if (count > EDEN_GOV_RECLAIMERS) count = EDEN_GOV_RECLAIMERS;
  for (unsigned i = 0; i < count && freed < want; ++i) {
    const eden_governor_reclaimer reclaimer = atomic_load_explicit(&eden_gov_reclaimers[i], memory_order_acquire);
    if (!reclaimer) continue;
    const size_t got = reclaimer(want - freed);
    atomic_fetch_add_explicit(&eden_gov_reclaimer_bytes[i], got, memory_order_relaxed);
    freed += got;
  }
  atomic_fetch_add_explicit(&eden_gov_reclaimed, freed, memory_order_relaxed);
  return freed;
}

/* Waits up to `max_ms` for the pool to have a free block of `bytes`, asking the reclaimers every
 * 100 ms. 1: there is one now (the caller asks again; another thread may take it first). 0: none
 * came, or max_ms is 0. Holds no lock of its own while it waits: the caller must not hold one the
 * reclaimers or the caches need. */
int eden_governor_wait(size_t bytes, unsigned max_ms, int owner) {
  if (max_ms == 0) return 0;
  const unsigned who = owner >= 0 && owner < EDEN_OWNERS ? (unsigned)owner : EDEN_OWNER_OTHER;
  atomic_fetch_add_explicit(&eden_gov_waiters, 1, memory_order_relaxed);
  const unsigned long long number = atomic_fetch_add_explicit(&eden_gov_waits, 1, memory_order_relaxed) + 1;
  atomic_fetch_add_explicit(&eden_gov_waits_by[who], 1, memory_order_relaxed);
  const uint64_t began = eden_gov_now_ns();
  uint64_t reclaimed_at = 0;
  size_t freed = 0, largest = 0;
  int won = 0;
  for (;;) {
    const uint64_t now = eden_gov_now_ns();
    if (reclaimed_at == 0 || now - reclaimed_at >= 100000000ull) {
      reclaimed_at = now;
      freed += eden_governor_reclaim(bytes);
    }
    largest = eden_governor_largest_free(1);
    if (largest >= bytes) {
      won = 1;
      break;
    }
    if (now - began >= (uint64_t)max_ms * 1000000ull) break;
    sceKernelUsleep(10000);
  }
  const unsigned long long ms = (eden_gov_now_ns() - began) / 1000000ull;
  atomic_fetch_sub_explicit(&eden_gov_waiters, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&eden_gov_wait_ms, ms, memory_order_relaxed);
  atomic_fetch_add_explicit(won ? &eden_gov_won : &eden_gov_lost, 1, memory_order_relaxed);
  if (number <= 32 || (number & 63) == 0 || !won)
    fprintf(stderr, "EDEN_GOV wait %llu for %zu KiB (%s): %s after %llu ms, reclaimers gave back %zu MiB, largest free %zu MiB\n",
            number, bytes >> 10, eden_owner_names[who], won ? "room" : "none", ms, freed >> 20, largest >> 20);
  return won;
}

/* How long the waits at a site may take, in ms (env.txt `name`, else `fallback`). */
unsigned eden_governor_wait_ms(const char *name, unsigned fallback) {
  return (unsigned)eden_gov_env_mib(name, fallback);
}

/* One line in stderr.log: the pool by owner, and what the governor did (headless/performance.cpp,
 * every 20 s, and every 5 s with EDEN_GOV_CENSUS=1). */
void eden_governor_report(void) {
  unsigned long long mib[EDEN_OWNERS];
  for (unsigned i = 0; i < EDEN_OWNERS; ++i) mib[i] = atomic_load_explicit(&eden_owner_bytes[i], memory_order_relaxed) >> 20;
  fprintf(stderr, "EDEN_GOV level %d, largest free %zu MiB; by owner: guest %llu heap %llu driver %llu jit %llu tables %llu "
          "stacks %llu other %llu MiB (untracked returns %llu MiB); waits %llu (tables %llu, buffers %llu): "
          "%llu got room, %llu none, %llu ms in all; reclaimers gave back %llu MiB:",
          eden_governor_level(), eden_governor_largest_free(1) >> 20, mib[EDEN_OWNER_GUEST], mib[EDEN_OWNER_HEAP],
          mib[EDEN_OWNER_DRIVER], mib[EDEN_OWNER_JIT], mib[EDEN_OWNER_TABLES], mib[EDEN_OWNER_STACK], mib[EDEN_OWNER_OTHER],
          atomic_load_explicit(&eden_ledger_untracked, memory_order_relaxed) >> 20,
          atomic_load_explicit(&eden_gov_waits, memory_order_relaxed),
          atomic_load_explicit(&eden_gov_waits_by[EDEN_OWNER_TABLES], memory_order_relaxed),
          atomic_load_explicit(&eden_gov_waits_by[EDEN_OWNER_DRIVER], memory_order_relaxed),
          atomic_load_explicit(&eden_gov_won, memory_order_relaxed), atomic_load_explicit(&eden_gov_lost, memory_order_relaxed),
          atomic_load_explicit(&eden_gov_wait_ms, memory_order_relaxed),
          atomic_load_explicit(&eden_gov_reclaimed, memory_order_relaxed) >> 20);
  unsigned count = atomic_load_explicit(&eden_gov_reclaimer_count, memory_order_relaxed);
  if (count > EDEN_GOV_RECLAIMERS) count = EDEN_GOV_RECLAIMERS;
  for (unsigned i = 0; i < count; ++i)
    if (eden_gov_reclaimer_names[i])
      fprintf(stderr, " %s %llu", eden_gov_reclaimer_names[i],
              atomic_load_explicit(&eden_gov_reclaimer_bytes[i], memory_order_relaxed) >> 20);
  fputc('\n', stderr);
}
