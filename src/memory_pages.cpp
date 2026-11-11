// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <new>
#include <sched.h>
#include <time.h>
#include <signal.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "fastmem.h"

// EDEN_UP_* switches (guest_memory_service.inc). The host checks compile this file without the
// core's include path (tools/check-jit-allocator.py): there, the same rule written out.
#if __has_include("common/pe_switch.h")
#include "common/pe_switch.h"
#else
namespace Common {
inline bool PeSwitch(const char* name) noexcept {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}
} // namespace Common
#endif

#if __has_include(<lz4.h>)
#include <lz4.h>
#define EDEN_SWAP_LZ4 1
#endif

#ifdef PS5_NATIVE
// Keep CPU heaps/tables and JIT views out of RADV's high-word-2 GPU window.
// This is a non-fixed hint: the kernel retains ownership of collision handling.
constexpr std::uintptr_t cpu_mapping_hint = 0x1000000000ull;
// Where the guest's own memory is mapped: above every address a guest can have (2^39, the end of
// the 39-bit address space of a 64-bit title). A guest page table entry is the page's host
// address less its guest address, and zero means "no pointer" (common/page_table.h). At the
// general hint the guest's memory sat at 67 to 71 GiB, inside the guest's address space: a game
// that mapped 2 MiB there at the matching offset got 512 entries of zero ("memory mapping base
// yield a nullptr within the table"), read zeros from that memory, jumped to address 0 and was
// restarted as a guest fault. A hint, as the other is: where the system places the block instead
// is reported (EDEN_GUEST_MEMORY).
constexpr std::uintptr_t guest_memory_hint = 0x8000000000ull;
static bool cpu_mapping_address(void* base) {
    return reinterpret_cast<std::uintptr_t>(base) >= 0x300000000ull && base != MAP_FAILED;
}
// A reserved range must also end below the driver's device memory at 0x40_0000_0000.
static bool cpu_mapping_range(void* base, std::size_t size) {
    const auto start = reinterpret_cast<std::uintptr_t>(base);
    return cpu_mapping_address(base) && size <= 0x4000000000ull && start <= 0x4000000000ull - size;
}
extern "C" {
std::int64_t sceKernelGetDirectMemorySize();
std::int32_t sceKernelAllocateDirectMemory(std::int64_t, std::int64_t, std::size_t,
                                         std::size_t, int, std::int64_t*);
std::int32_t sceKernelMapDirectMemory(void**, std::size_t, int, int, std::int64_t, std::size_t);
std::int32_t sceKernelReleaseDirectMemory(std::int64_t, std::size_t);
std::int32_t sceKernelAvailableDirectMemorySize(std::int64_t, std::int64_t, std::size_t, std::int64_t*, std::size_t*);
std::int32_t sceKernelReserveVirtualRange(void**, std::size_t, int, std::size_t);
std::int32_t sceKernelEnableDmemAliasing();
int sceKernelDebugOutText(int, const char*);
}
#endif

// Threads of the guest memory service name themselves for the crash report and the thread dump
// (headless/crash_report.cpp). Weak: a check built without the frontend links, and skips it.
extern "C" void eden_crash_name_thread(const char* name) __attribute__((weak));

namespace Common {
namespace {
// The header sits in the page before the data. Blocks of at least LargePage start their data
// one LargePage in, with the address and direct memory LargePage-aligned, so the kernel can map
// them with 2 MiB pages: the guest backing, JIT caches and page tables are walked on every
// guest memory access, and 16 KiB pages cover only a few MiB of TLB reach.
struct Header { std::int64_t physical; std::size_t total; std::size_t lead; };
constexpr std::size_t LargePage = 0x200000;
// Development A/B: dev-settings large_pages=off keeps every block 16 KiB-aligned. The heap takes
// its blocks before the frontend parses the file, so read it here with plain system calls.
bool DevSetting(const char* entry) {
    char text[4096];
    const int fd = open("/app0/dev-settings.txt", O_RDONLY);
    if (fd < 0) return false;
    const auto count = read(fd, text, sizeof(text) - 1);
    close(fd);
    text[count > 0 ? count : 0] = '\0';
    return std::strstr(text, entry) != nullptr;
}
// A line for the log from inside the allocator: no stdio (it may allocate). On the console the
// app's stderr is a stream of the C library and not descriptor 2, so the line goes to the kernel
// log.
void Note(const char* line) noexcept {
#ifdef PS5_NATIVE
    sceKernelDebugOutText(0, line);
#else
    (void)!write(2, line, std::strlen(line));
#endif
}
// The same for a line written outside the allocator: on the console it is in the app's log too.
void Report(const char* line) noexcept {
    Note(line);
#ifdef PS5_NATIVE
    std::fputs(line, stderr);
#endif
}
bool LargePagesEnabled() {
    static std::atomic<int> state{0}; // 0 unknown, 1 on, 2 off
    int value = state.load(std::memory_order_acquire);
    if (value == 0) {
        value = DevSetting("large_pages=off") ? 2 : 1;
        state.store(value, std::memory_order_release);
    }
    return value == 1;
}
std::size_t lead_size(std::size_t size, std::size_t page) {
    return size >= LargePage && LargePagesEnabled() ? LargePage : page;
}
std::size_t allocation_size(std::size_t size, std::size_t page, std::size_t lead) {
    if (!size || page < sizeof(Header) || size > std::numeric_limits<std::size_t>::max() - 2 * lead)
        return 0;
    return (size + lead - 1) / lead * lead + lead;
}
const Header& header_of(const void* pointer, std::size_t page) {
    return *reinterpret_cast<const Header*>(static_cast<const std::uint8_t*>(pointer) - page);
}
}

// PS5 direct memory is real memory from the first byte: there is no page the system fills in
// when it is first touched. A block from here is dense and zeroed. What must not be dense (the
// guest page tables, the heap) uses the two families below.
namespace {
void* AllocatePagesAt(std::size_t size, [[maybe_unused]] std::uintptr_t hint,
                      [[maybe_unused]] bool retry_with_small_pages = false) noexcept {
    const long page = sysconf(_SC_PAGESIZE);
    std::size_t lead = page > 0 ? lead_size(size, static_cast<std::size_t>(page)) : 0;
    auto total = page > 0 ? allocation_size(size, page, lead) : 0;
    if (!total) { errno = EINVAL; return nullptr; }
    void* base = nullptr;
    std::int64_t physical = -1;
#ifdef PS5_NATIVE
    base = reinterpret_cast<void*>(hint);
    const auto limit = sceKernelGetDirectMemorySize();
    auto rc = sceKernelAllocateDirectMemory(0, limit, total, lead, 12, &physical);
    if (rc != 0 && retry_with_small_pages && lead == LargePage) {
        // A medium heap block can fit without the large-page alignment and padding.
        lead = static_cast<std::size_t>(page);
        total = allocation_size(size, lead, lead);
        rc = sceKernelAllocateDirectMemory(0, limit, total, lead, 12, &physical);
    }
    if (rc != 0) {
        std::fprintf(stderr, "Direct allocation failed: rc=%08x bytes=%zu limit=%lld\n",
                     unsigned(rc), total, static_cast<long long>(limit));
        errno = ENOMEM;
        return nullptr;
    }
    rc = sceKernelMapDirectMemory(&base, total, PROT_READ | PROT_WRITE, 0, physical, lead);
    if (rc != 0 || !cpu_mapping_address(base)) {
        std::fprintf(stderr, "Direct mapping failed: rc=%08x bytes=%zu\n", unsigned(rc), total);
        if (rc == 0 && base && base != MAP_FAILED && munmap(base, total) != 0) std::abort();
        if (sceKernelReleaseDirectMemory(physical, total) != 0) std::abort();
        errno = ENOMEM;
        return nullptr;
    }
    if (lead == LargePage) {
        // The heap allocates through here: no stdio (it may allocate).
        char line[96];
        const int length = std::snprintf(line, sizeof(line), "EDEN_LARGE_ALLOC bytes=%zu va=%p pa=%llx\n",
                                         total, base, static_cast<unsigned long long>(physical));
        if (length > 0) Note(line);
    }
    std::memset(base, 0, total);
#else
    base = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) return nullptr;
#endif
    auto* data = static_cast<std::uint8_t*>(base) + lead;
    new (data - page) Header{physical, total, lead};
    return data;
}
} // namespace

void* AllocateMemoryPages(std::size_t size) noexcept {
#ifdef PS5_NATIVE
    return AllocatePagesAt(size, cpu_mapping_hint);
#else
    return AllocatePagesAt(size, 0);
#endif
}

void* AllocateHeapMemoryPages(std::size_t size) noexcept {
#ifdef PS5_NATIVE
    return AllocatePagesAt(size, cpu_mapping_hint, size < 32 * 1024 * 1024);
#else
    return AllocateMemoryPages(size);
#endif
}

// The guest's memory (host_memory.cpp): as AllocateMemoryPages, above the guest's address space.
void* AllocateGuestMemory(std::size_t size) noexcept {
#ifdef PS5_NATIVE
    void* base = AllocatePagesAt(size, guest_memory_hint);
    if (!base) base = AllocatePagesAt(size, cpu_mapping_hint);
    char line[112];
    const int length = std::snprintf(line, sizeof(line), "EDEN_GUEST_MEMORY va=%p bytes=%zu above_guest=%d\n", base, size,
                                     int(reinterpret_cast<std::uintptr_t>(base) >= guest_memory_hint));
    if (length > 0) Report(line);
    return base;
#else
    return AllocatePagesAt(size, 0);
#endif
}

#ifdef PS5_NATIVE
// Direct-memory start of an AllocateMemoryPages block's first data byte.
std::int64_t DirectMemoryStart(const void* pointer) noexcept {
    const long page = sysconf(_SC_PAGESIZE);
    if (!pointer || page <= 0) std::abort();
    const auto& header = header_of(pointer, page);
    return header.physical + static_cast<std::int64_t>(header.lead);
}

// A second view of our own direct allocation; ownership stays with pointer.
void* MapExecutableAlias(void* pointer, std::size_t size) noexcept {
    const long page = sysconf(_SC_PAGESIZE);
    if (!pointer || page <= 0) return nullptr;
    const auto header = header_of(pointer, page);
    if (header.total != allocation_size(size, page, header.lead)) std::abort();
    // This wrapper reports zero unconditionally; the mapping is the actual check.
    static const auto enabled = sceKernelEnableDmemAliasing();
    (void)enabled;
    const auto span = header.total - header.lead;
    void* alias = reinterpret_cast<void*>(cpu_mapping_hint);
    const auto rc = sceKernelMapDirectMemory(&alias, span, PROT_READ, 0,
                                              header.physical + static_cast<std::int64_t>(header.lead),
                                              header.lead);
    if (rc != 0 || !cpu_mapping_address(alias)) {
        std::printf("EDEN_JIT_ALIAS_MAP rc=%08x bytes=%zu\n", unsigned(rc), span);
        if (rc == 0 && alias && alias != MAP_FAILED && munmap(alias, span) != 0) std::abort();
        errno = rc ? unsigned(rc) & 0xffff : ENOMEM;
        return nullptr;
    }
    // Direct mapping wrappers reject EXEC on some versions. Use the same
    // checked mprotect contract already qualified for this owned memory type.
    if (mprotect(alias, span, PROT_READ | PROT_EXEC) != 0) {
        const auto error = errno;
        if (munmap(alias, span) != 0) std::abort();
        errno = error;
        return nullptr;
    }
    return alias;
}
#endif

void FreeMemoryPages(void* pointer, std::size_t size) noexcept {
    if (!pointer) return;
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) std::abort();
    const auto header = header_of(pointer, page);
    auto* base = static_cast<std::uint8_t*>(pointer) - header.lead;
    if (header.total != allocation_size(size, page, header.lead) || munmap(base, header.total) != 0) std::abort();
#ifdef PS5_NATIVE
    if (sceKernelReleaseDirectMemory(header.physical, header.total) != 0) std::abort();
#endif
}
// Xbyak's allocator interface supplies only the pointer at release. The owned
// header already records the exact mapped size; retain the same checked free.
void FreeMemoryPages(void* pointer) noexcept {
    if (!pointer) return;
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) std::abort();
    const auto header = header_of(pointer, page);
    if (header.total <= header.lead) std::abort();
    FreeMemoryPages(pointer, header.total - header.lead);
}

// ---- One block kept for a big image: EDEN_BIG_IMAGE_RESERVE_MIB ----
//
// Ocarina of Time's 6144x6144 ground texture is decoded on the CPU into one 144 MiB buffer
// (video_core/texture_cache/texture_cache.h, QueueAsyncDecode: decoded_data.resize_destructive).
// Minutes into a game the pool has no single free range that large ("Direct allocation failed:
// rc=80020023 bytes=153092096", with 300 to 514 MiB free in pieces), so the heap's large block
// took its memory in pieces (eden_heap_split_pages) and left the pool with 2 MiB; the pipeline
// builder threads' next allocations found nothing and the session ended (SIGSEGV on
// VkPipelineBuilder, 7 Nov). With the switch set, one block of n MiB is taken from the top of
// the pool while the pool is still whole (headless/heap_pages.cpp, eden_heap_pool_init, from
// main after env.txt), and goes only to a large block that found no single free range and needs
// at least 7/8 of it; it is back in the reserve when that block is freed. One holder at a time:
// a second such request goes in pieces as before, or fails with EDEN_BIG_IMAGE_NO_SPLIT=1
// (headless/heap_pages.cpp).
#ifdef PS5_NATIVE
extern "C" int eden_direct_high(std::size_t size, std::size_t alignment, int type, std::int64_t* start);
#endif
namespace {
#ifdef PS5_NATIVE
struct BigImageReserve {
    std::atomic<int> state{0}; // 0: none; 3: being taken; 1: free; 2: given out
    void* base = nullptr;
    std::size_t size = 0;
    std::int64_t physical = -1;
    // The reserve's class: requests of [least, most] bytes (7/8 of n MiB to n MiB, rounded up to whole
    // large pages). Ocarina's runs also ask for 82 to 108 MiB blocks at the same area loads (stderr
    // "Direct allocation failed": 88080384, 102760448, 115343360 bytes) and 216 MiB ones (228589568):
    // with a class of n/2 the first could hold the reserve when the 144 MiB block comes, and the second
    // could never be served but was refused by EDEN_BIG_IMAGE_NO_SPLIT for good. (Review, 8 Nov.)
    std::atomic<std::size_t> least{0}, most{0};
    std::atomic<unsigned> given{0}, refused{0};
};
BigImageReserve big_image_reserve;
#endif
} // namespace

// A request of the reserve's class: 7/8 of the size asked for with EDEN_BIG_IMAGE_RESERVE_MIB up to that
// size, whether or not the reserve could be taken. False with the switch unset.
bool BigImageRequest(std::size_t bytes) noexcept {
#ifdef PS5_NATIVE
    const std::size_t least = big_image_reserve.least.load(std::memory_order_relaxed);
    const std::size_t most = big_image_reserve.most.load(std::memory_order_relaxed);
    return least != 0 && bytes >= least && bytes <= most;
#else
    (void)bytes;
    return false;
#endif
}

// Takes the reserve: a block of mib MiB (rounded up to whole large pages). Once; later calls do
// nothing. Reports what it did (EDEN_BIG_IMAGE_RESERVE).
void TakeBigImageReserve(std::size_t mib) noexcept {
#ifdef PS5_NATIVE
    auto& reserve = big_image_reserve;
    if (mib == 0 || mib > 4096) return;
    int expected = 0;
    if (!reserve.state.compare_exchange_strong(expected, 3, std::memory_order_acq_rel)) return;
    const std::size_t size = ((mib << 20) + LargePage - 1) / LargePage * LargePage;
    reserve.most.store(size, std::memory_order_relaxed);
    reserve.least.store(size - size / 8, std::memory_order_relaxed);
    std::int64_t physical = -1;
    char line[160];
    if (eden_direct_high(size, LargePage, 12, &physical) != 0) {
        std::snprintf(line, sizeof(line), "EDEN_BIG_IMAGE_RESERVE none: the pool has no free range of %zu MiB\n",
                      size >> 20);
        Report(line);
        reserve.state.store(0, std::memory_order_release);
        return;
    }
    void* base = reinterpret_cast<void*>(cpu_mapping_hint);
    const int rc = sceKernelMapDirectMemory(&base, size, PROT_READ | PROT_WRITE, 0, physical, LargePage);
    if (rc != 0 || !cpu_mapping_range(base, size)) {
        if (rc == 0 && base != nullptr && base != MAP_FAILED) (void)munmap(base, size);
        (void)sceKernelReleaseDirectMemory(physical, size);
        std::snprintf(line, sizeof(line), "EDEN_BIG_IMAGE_RESERVE none: mapping failed rc=%08x bytes=%zu\n",
                      unsigned(rc), size);
        Report(line);
        reserve.state.store(0, std::memory_order_release);
        return;
    }
    reserve.base = base;
    reserve.size = size;
    reserve.physical = physical;
    reserve.state.store(1, std::memory_order_release);
    std::snprintf(line, sizeof(line), "EDEN_BIG_IMAGE_RESERVE %zu MiB va=%p pa=%llx\n", size >> 20, base,
                  static_cast<unsigned long long>(physical));
    Report(line);
#else
    (void)mib;
#endif
}

// The reserve for a large block of `bytes` (zeroed, as the heap's other large blocks), or nullptr:
// no reserve, given out already, too small for the request, or the request under 7/8 of it (a
// smaller block has room enough in the pool's pieces, and must not hold the reserve when the
// big image comes). Called inside malloc: no stdio.
void* TakeBigImageBlock(std::size_t bytes) noexcept {
#ifdef PS5_NATIVE
    auto& reserve = big_image_reserve;
    if (reserve.state.load(std::memory_order_acquire) != 1 || bytes == 0) return nullptr;
    if (bytes > reserve.size || bytes < reserve.least.load(std::memory_order_relaxed)) return nullptr;
    int expected = 1;
    if (!reserve.state.compare_exchange_strong(expected, 2, std::memory_order_acq_rel)) {
        reserve.refused.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    std::memset(reserve.base, 0, bytes);
    const unsigned given = reserve.given.fetch_add(1, std::memory_order_relaxed) + 1;
    char line[128];
    const int length = std::snprintf(line, sizeof(line), "EDEN_BIG_IMAGE_RESERVE given bytes=%zu va=%p times=%u\n",
                                     bytes, reserve.base, given);
    if (length > 0) Note(line);
    return reserve.base;
#else
    (void)bytes;
    return nullptr;
#endif
}

// 1: the block was the reserve, which is free again; 0: some other block.
int GiveBigImageBlock(void* address) noexcept {
#ifdef PS5_NATIVE
    auto& reserve = big_image_reserve;
    if (address == nullptr || address != reserve.base ||
        reserve.state.load(std::memory_order_acquire) != 2)
        return 0;
    reserve.state.store(1, std::memory_order_release);
    return 1;
#else
    (void)address;
    return 0;
#endif
}

// ---- Sparse pages: Eden's large tables ----
//
// Eden indexes its guest page table (and the GPU's address tables) directly: 8 bytes for every
// 4 KiB of a 512 GiB address space is a 1 GiB table, of which a game writes a few MiB. On a PC
// that costs nothing: the table is mapped read-only, the system shows one shared page of zeroes
// wherever nothing was written, and SparseLargeVector makes a page writable before its first
// write. The PS5 has no such page, so the same thing is built from mappings:
//
//   - the table's address range is reserved;
//   - every 2 MiB slot of it shows one shared block of zeroes, read-only (the JIT reads the
//     table for any address a game touches, mapped or not);
//   - CommitSparsePage, called before a first write, gives the slot a zeroed block of its own.
//
// A session's page table then takes 2 MiB per GiB of address space the game maps, not 1 GiB.
// The Linux build mirrors it (read-only until committed), so the host checks fail on a write
// that was not announced, as the console would.
namespace {
constexpr std::size_t SparseSlot = LargePage;
struct SparseRange {
    std::atomic<std::uintptr_t> begin{0}; // 0: free
    std::uintptr_t end = 0;
    std::atomic<bool>* owned = nullptr; // per slot: it has a block of its own
    bool guest = false;                 // the guest's memory, not a table (counted apart)
    std::int64_t* physical = nullptr;   // that block (PS5)
    std::atomic<std::uint8_t>* state = nullptr; // per slot, when the range is a service (guest_memory_service.inc)
    bool promote = false;               // writes to shared slots are promoted (the service runs)
    std::uint16_t* fine = nullptr;      // per slot: 16 KiB pages a Fine slot keeps (low 8 bits), times made whole (bits 8-11)
    std::uint32_t* fine_after = nullptr; // per slot: the scan pass from which it may be made Fine again
    std::uint32_t* swap = nullptr;      // per slot: the tick it was last touched (Own) or armed (Probe), its place in the file (Out)
};
std::array<SparseRange, 64> sparse_ranges;
std::mutex sparse_mutex;
std::atomic<std::size_t> sparse_reserved{0}, sparse_committed{0};
// The guest's memory when it is such a range (below: "The guest's memory, backed where it is used").
std::atomic<std::size_t> guest_reserved{0}, guest_committed{0};
std::atomic<SparseRange*> guest_range{nullptr};
std::atomic<int> sparse_state{0}; // 0 unknown, 1 in use, 2 not available (tables are dense)

SparseRange* SparseRangeOf(std::uintptr_t address) {
    for (auto& range : sparse_ranges) {
        const auto begin = range.begin.load(std::memory_order_acquire);
        if (begin != 0 && address >= begin && address < range.end) return &range;
    }
    return nullptr;
}

#ifdef PS5_NATIVE
// From as high in the pool as there is room (src/direct_memory.c): what is committed this way,
// the heap's pieces, stays for good, while a game's memory and its graphics come and go below.
extern "C" int eden_direct_high(std::size_t size, std::size_t alignment, int type, std::int64_t* start);
// Where the system puts it: a 2 MiB block of this file's, which a session gives back.
extern "C" int eden_direct_low(std::size_t size, std::size_t alignment, int type, std::int64_t* start);
extern "C" int eden_direct_guest(std::size_t size, std::size_t alignment, int type, std::int64_t* start);
// The pool governor (src/direct_memory.c): the same with the owner named for its ledger, and the wait
// for memory a request the pool refused may take instead of ending the session. Weak: the host check
// of this file (tools/check-jit-allocator.py) links without them.
extern "C" int eden_direct_owned(int owner, int high, std::size_t size, std::size_t alignment, int type,
                                 std::int64_t* start) __attribute__((weak));
extern "C" int eden_governor_wait(std::size_t bytes, unsigned max_ms, int owner) __attribute__((weak));
extern "C" unsigned eden_governor_wait_ms(const char* name, unsigned fallback) __attribute__((weak));
#endif

namespace {
// The governor's owners (src/direct_memory.c, EDEN_OWNER_*).
enum : int { GovernorJit = 4, GovernorTables = 5, GovernorHeap = 2 };
#ifdef PS5_NATIVE
int DirectOwned(int owner, bool high, std::size_t size, std::size_t alignment, std::int64_t* start) {
    if (eden_direct_owned) return eden_direct_owned(owner, high ? 1 : 0, size, alignment, 12, start);
    return high ? eden_direct_high(size, alignment, 12, start) : eden_direct_low(size, alignment, 12, start);
}
#endif
} // namespace

#ifdef PS5_NATIVE
std::int64_t zero_block = -1;

// Where a slot's short-lived view of a block goes. A view asks for the first gap at or after the
// address, and at cpu_mapping_hint that can be the gap a heap piece leaves between DecommitMemoryRange's
// munmap and its reservation, MAP_FIXED, which then takes the view's mapping away. These views are
// made in the fault handler, so the clear that follows faults inside it (run 59's analysis found this
// route; no run has shown it). EDEN_SLOT_VIEW_HIGH=1 (ServiceStart) puts them high above the heap.
std::atomic<std::uintptr_t> slot_view_hint{cpu_mapping_hint};
constexpr std::uintptr_t slot_view_high = 0x3800000000ull;

// A 2 MiB block of direct memory, zeroed through a mapping of its own that is gone again.
bool ZeroedBlock(std::int64_t* physical, bool guest = false) {
    if (guest ? eden_direct_guest(SparseSlot, SparseSlot, 12, physical) != 0
              : DirectOwned(GovernorTables, false, SparseSlot, SparseSlot, physical) != 0)
        return false;
    void* view = reinterpret_cast<void*>(slot_view_hint.load(std::memory_order_relaxed));
    if (sceKernelMapDirectMemory(&view, SparseSlot, PROT_READ | PROT_WRITE, 0, *physical, SparseSlot) != 0) {
        (void)sceKernelReleaseDirectMemory(*physical, SparseSlot);
        return false;
    }
    std::memset(view, 0, SparseSlot);
    if (munmap(view, SparseSlot) != 0) std::abort();
    return true;
}
bool MapSlot(std::uintptr_t at, int protection, std::int64_t physical) {
    void* address = reinterpret_cast<void*>(at);
    return sceKernelMapDirectMemory(&address, SparseSlot, protection, MAP_FIXED, physical, SparseSlot) == 0 &&
           address == reinterpret_cast<void*>(at);
}
#endif
} // namespace

#ifdef PS5_NATIVE
// A mapping of ours becomes reserved address space in one call: a fixed reservation replaces what
// it covers. Unmapped first and reserved after, the range was free for a moment, and a mapping the
// system placed meanwhile (ZeroedBlock's view, any other without a fixed address) was overwritten
// by the reservation: an experimental build that gave heap memory back often (mimalloc,
// docs/MEMORY-CPU-EXPERIMENTS.md) faulted so in FillReserve's memset. Refused, it goes the old way.
// 1: reserved; 0: unmapped, not reserved again (the range must not be used again); -1: still mapped
// (nothing was done).
int ReserveOver(void* at, std::size_t size, std::size_t alignment) noexcept {
    void* again = at;
    if (sceKernelReserveVirtualRange(&again, size, MAP_FIXED, alignment) == 0 && again == at) return 1;
    if (munmap(at, size) != 0) return -1;
    again = at;
    return sceKernelReserveVirtualRange(&again, size, MAP_FIXED, alignment) == 0 && again == at ? 1 : 0;
}
#endif

namespace {

// Reserves `span` bytes whose every slot reads as zeroes. Null when the platform refuses. The
// guest's memory goes above the guest's address space, as it does when it is taken whole.
void* ReserveSparse(std::size_t span, [[maybe_unused]] bool guest = false) {
#ifdef PS5_NATIVE
    void* address = reinterpret_cast<void*>(guest ? guest_memory_hint : cpu_mapping_hint);
    if (sceKernelReserveVirtualRange(&address, span, 0, SparseSlot) != 0) return nullptr;
    if (guest ? reinterpret_cast<std::uintptr_t>(address) < guest_memory_hint : !cpu_mapping_range(address, span)) {
        (void)munmap(address, span);
        return nullptr;
    }
    const auto start = reinterpret_cast<std::uintptr_t>(address);
    for (std::size_t offset = 0; offset < span; offset += SparseSlot) {
        if (!MapSlot(start + offset, PROT_READ, zero_block)) {
            (void)munmap(address, span);
            return nullptr;
        }
    }
    return address;
#else
    void* address = mmap(nullptr, span, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return address == MAP_FAILED ? nullptr : address;
#endif
}

// Gives the slot at `at` memory of its own, zeroed and writable. sparse_mutex is held.
#ifdef PS5_NATIVE
bool TableReserveBlock(std::int64_t* physical) noexcept; // guest_memory_service.inc
#endif
bool OwnSlot(std::uintptr_t at, std::int64_t* physical, [[maybe_unused]] bool guest = false) {
#ifdef PS5_NATIVE
    // The block is zeroed before it appears in the table: a JIT thread may read the slot at any
    // moment, and the mapping changes in one step.
    // A table slot has nowhere to go if this fails (CommitSparsePage ends the app), so it may
    // take a block of the guest reserve, as the guest writes do: run 62 ended here with 242 MiB
    // of the reserve idle.
    if (!ZeroedBlock(physical, guest) && !TableReserveBlock(physical)) return false;
    if (MapSlot(at, PROT_READ | PROT_WRITE, *physical)) return true;
    (void)sceKernelReleaseDirectMemory(*physical, SparseSlot);
    return false;
#else
    *physical = 0;
    return mprotect(reinterpret_cast<void*>(at), SparseSlot, PROT_READ | PROT_WRITE) == 0;
#endif
}

// The service's side of a range (guest_memory_service.inc).
void ReleaseServiceBlocks(SparseRange& range) noexcept;
bool ServiceCommit(SparseRange& range, std::size_t slot) noexcept;
void ServiceStart(SparseRange& range) noexcept;
void ServiceStop(SparseRange& range) noexcept;
bool InCodeCache(std::uintptr_t address) noexcept; // code the JIT made (below, with the code ranges)

void ReleaseSparse(SparseRange& range) {
    if (range.state != nullptr) ReleaseServiceBlocks(range); // its blocks, shared ones once
    const auto begin = range.begin.load(std::memory_order_relaxed);
    const std::size_t span = range.end - begin;
    if (munmap(reinterpret_cast<void*>(begin), span) != 0) std::abort();
    std::size_t owned = 0;
    for (std::size_t slot = 0; slot < span / SparseSlot; ++slot) {
        if (!range.owned[slot].load(std::memory_order_relaxed)) continue;
        ++owned;
#ifdef PS5_NATIVE
        if (sceKernelReleaseDirectMemory(range.physical[slot], SparseSlot) != 0) std::abort();
#endif
    }
    (range.guest ? guest_reserved : sparse_reserved).fetch_sub(span, std::memory_order_relaxed);
    (range.guest ? guest_committed : sparse_committed).fetch_sub(owned * SparseSlot, std::memory_order_relaxed);
    range.guest = false;
    range.promote = false;
    delete[] range.state;
    range.state = nullptr;
    delete[] range.fine;
    range.fine = nullptr;
    delete[] range.fine_after;
    range.fine_after = nullptr;
    delete[] range.swap;
    range.swap = nullptr;
    range.begin.store(0, std::memory_order_release);
    delete[] range.owned;
    delete[] range.physical;
    range.owned = nullptr;
    range.physical = nullptr;
}

// Once: can tables be sparse here? On the console this is a check of the three mapping steps
// above with real memory; a refusal leaves the tables dense, as they were before.
bool SparseAvailable() {
    int state = sparse_state.load(std::memory_order_acquire);
    if (state != 0) return state == 1;
    const std::lock_guard lock{sparse_mutex};
    state = sparse_state.load(std::memory_order_relaxed);
    if (state != 0) return state == 1;
    bool ok = true;
    const char* step = "ok";
#ifdef PS5_NATIVE
    ok = !DevSetting("sparse_tables=off");
    if (!ok) step = "switched-off";
    if (ok) {
        (void)sceKernelEnableDmemAliasing(); // reports zero regardless; the mappings below are the check
        ok = ZeroedBlock(&zero_block);
        if (!ok) step = "zero-block";
    }
#endif
    if (ok) {
        auto* probe = static_cast<volatile std::uint64_t*>(ReserveSparse(2 * SparseSlot));
        const std::size_t words = SparseSlot / sizeof(std::uint64_t);
        std::int64_t physical = -1;
        if (probe == nullptr) {
            ok = false;
            step = "reserve";
        } else {
            if (probe[0] != 0 || probe[words - 1] != 0 || probe[words] != 0 || probe[2 * words - 1] != 0) {
                ok = false;
                step = "zero-read";
            } else if (!OwnSlot(reinterpret_cast<std::uintptr_t>(probe) + SparseSlot, &physical)) {
                ok = false;
                step = "own";
            } else {
                probe[words] = 0x5a17ed5a17ed5a17ull;
                probe[2 * words - 1] = 0x1234567890abcdefull;
                if (probe[words] != 0x5a17ed5a17ed5a17ull || probe[2 * words - 1] != 0x1234567890abcdefull ||
                    probe[0] != 0 || probe[words - 1] != 0 || probe[words + 1] != 0) {
                    ok = false;
                    step = "own-read";
                }
#ifdef PS5_NATIVE
                // The written slot goes back; the shared block must still be all zeroes.
                if (munmap(const_cast<std::uint64_t*>(probe) + words, SparseSlot) != 0 ||
                    sceKernelReleaseDirectMemory(physical, SparseSlot) != 0)
                    std::abort();
                if (!MapSlot(reinterpret_cast<std::uintptr_t>(probe) + SparseSlot, PROT_READ, zero_block) ||
                    probe[words] != 0 || probe[2 * words - 1] != 0) {
                    ok = false;
                    step = "shared-zero";
                }
#endif
            }
            if (munmap(const_cast<std::uint64_t*>(probe), 2 * SparseSlot) != 0) std::abort();
        }
    }
    char line[96];
    const int length = std::snprintf(line, sizeof(line), "EDEN_SPARSE_TABLES available=%d step=%s\n", ok ? 1 : 0, step);
    if (length > 0) Report(line);
    sparse_state.store(ok ? 1 : 2, std::memory_order_release);
    return ok;
}
} // namespace

// The first table asks this; the app asks at start, so the log says it in every session.
bool SparseTablesAvailable() noexcept {
    return SparseAvailable();
}

namespace {
// A sparse range of `size` bytes: every slot reads as zeroes until CommitSparsePage gives it a
// block of its own. Null when the size is too small for one or the platform refuses.
const char* sparse_refused = "";  // why the last sparse range was refused, for the log
void* NewSparseRange(std::size_t size, bool guest) {
    sparse_refused = "";
    const std::size_t span = (size + SparseSlot - 1) / SparseSlot * SparseSlot;
    if (size < 2 * SparseSlot || span < size) {
        sparse_refused = "too-small";
        return nullptr;
    }
    if (!SparseAvailable()) {
        sparse_refused = "platform";
        return nullptr;
    }
    const std::lock_guard lock{sparse_mutex};
    SparseRange* free_range = nullptr;
    for (auto& range : sparse_ranges)
        if (range.begin.load(std::memory_order_relaxed) == 0) {
            free_range = &range;
            break;
        }
    if (free_range == nullptr) {
        sparse_refused = "no-range";
        return nullptr;
    }
    const std::size_t slots = span / SparseSlot;
    auto* owned = free_range ? new (std::nothrow) std::atomic<bool>[slots] : nullptr;
    auto* physical = owned ? new (std::nothrow) std::int64_t[slots] : nullptr;
    if (physical == nullptr) {
        delete[] owned;
        sparse_refused = "no-table";
        return nullptr;
    }
    void* address = ReserveSparse(span, guest);
    if (address == nullptr) {
        sparse_refused = "reserve";
        delete[] owned;
        delete[] physical;
        return nullptr;
    }
    for (std::size_t slot = 0; slot < slots; ++slot) {
        owned[slot].store(false, std::memory_order_relaxed);
        physical[slot] = -1;
    }
    free_range->owned = owned;
    free_range->physical = physical;
    free_range->guest = guest;
    free_range->end = reinterpret_cast<std::uintptr_t>(address) + span;
    free_range->begin.store(reinterpret_cast<std::uintptr_t>(address), std::memory_order_release);
    (guest ? guest_reserved : sparse_reserved).fetch_add(span, std::memory_order_relaxed);
    if (guest) guest_range.store(free_range, std::memory_order_release);
    return address;
}
} // namespace

void* AllocateSparsePages(std::size_t size) noexcept {
    if (void* const address = NewSparseRange(size, false)) return address;
    if (size >= 2 * SparseSlot && SparseAvailable()) {
        // The platform took the check's two slots but not this table: it is dense, as before.
        char line[96];
        const int length = std::snprintf(line, sizeof(line), "EDEN_SPARSE_TABLES dense bytes=%zu\n", size);
        if (length > 0) Report(line);
    }
    return AllocateMemoryPages(size);
}

void FreeSparsePages(void* base, std::size_t size) noexcept {
    if (!base) return;
    {
        const std::lock_guard lock{sparse_mutex};
        SparseRange* range = SparseRangeOf(reinterpret_cast<std::uintptr_t>(base));
        if (range != nullptr) {
            if (range->begin.load(std::memory_order_relaxed) != reinterpret_cast<std::uintptr_t>(base)) std::abort();
            ReleaseSparse(*range);
            return;
        }
    }
    FreeMemoryPages(base, size); // a table that was allocated dense
}

// ---- The guest's memory, backed where it is used ----
//
// A game's console has 4 GiB, and the guest's kernel divides them into pools: the game's, the
// applets', the system's. Taken whole, the block is 4 GiB of the pool the CPU shares with the
// GPU from the moment a game starts. But the kernel hands a pool's pages out as they are asked
// for, and a page it has not handed out was never written: the applets' pool is not used by a
// game at all, most of the system's is not, and a small game leaves most of its own. So the
// block is a sparse range like the tables above, each 2 MiB slot one shared block of zeroes
// until it is committed, and it is committed where the guest's kernel uses it: everything
// outside the pools when the kernel starts, and a pool's pages when they are handed out (the
// two places are in headless/CMakeLists.txt, "The guest's memory is backed where it is used").
// Reading a slot that was never committed gives zeroes, as the block taken whole would;
// writing one that was not announced stops the app, on the console as in the host checks.
void CommitSparsePage(std::uintptr_t page) noexcept;

void* AllocateGuestMemoryWhereUsed(std::size_t size) noexcept {
    // Development A/B: dev-settings guest_memory=whole takes the block whole, as before.
    const bool whole = DevSetting("guest_memory=whole");
    void* base = whole ? nullptr : NewSparseRange(size, true);
    if (base == nullptr) {
        // The block taken whole is `size` of the pool the CPU shares with the GPU, at the moment
        // the game starts. Say which step refused the block backed where used, so a layout above
        // the Switch's own 4 GiB that will not fit is one log line and not a session that dies
        // somewhere else.
        const char* why = whole ? "asked-for" : sparse_refused;
        char line[128];
        const int length = std::snprintf(line, sizeof(line),
                                         "EDEN_GUEST_MEMORY sparse refused bytes=%zu step=%s\n", size,
                                         why && *why ? why : "unknown");
        if (length > 0) Report(line);
        return AllocateGuestMemory(size);
    }
    char line[112];
    const int length = std::snprintf(line, sizeof(line), "EDEN_GUEST_MEMORY va=%p bytes=%zu above_guest=1 backed=where-used\n",
                                     base, size);
    if (length > 0) Report(line);
    if (SparseRange* const range = guest_range.load(std::memory_order_acquire)) ServiceStart(*range);
    return base;
}

// Memory behind `bytes` of the guest's memory from `offset` on (whole slots). Nothing to do for
// a block that was taken whole.
void CommitGuestMemory(std::uint64_t offset, std::size_t bytes) noexcept {
    SparseRange* const range = guest_range.load(std::memory_order_acquire);
    if (range == nullptr || bytes == 0) return;
    const auto begin = range->begin.load(std::memory_order_relaxed);
    const std::size_t span = range->end - begin;
    if (offset >= span) return;
    if (bytes > span - offset) bytes = span - static_cast<std::size_t>(offset);
    const std::size_t last = (static_cast<std::size_t>(offset) + bytes - 1) / SparseSlot;
    for (std::size_t slot = static_cast<std::size_t>(offset) / SparseSlot; slot <= last; ++slot)
        if (!range->owned[slot].load(std::memory_order_acquire)) CommitSparsePage(begin + slot * SparseSlot);
}

// The guest's memory, however it was taken.
void FreeGuestMemory(void* base, std::size_t size) noexcept {
    if (SparseRange* const range = guest_range.load(std::memory_order_acquire);
        range != nullptr && range->begin.load(std::memory_order_relaxed) == reinterpret_cast<std::uintptr_t>(base)) {
        char line[112];
        const int length = std::snprintf(line, sizeof(line), "EDEN_GUEST_MEMORY given back: held=%zu of=%zu\n",
                                         guest_committed.load(std::memory_order_relaxed), size);
        if (length > 0) Report(line);
        ServiceStop(*range);
        guest_range.store(nullptr, std::memory_order_release);
    }
    FreeSparsePages(base, size);
}

// The span of the guest's memory and the memory behind it; both 0 for a block taken whole.
void GuestMemoryUsage(std::size_t* span, std::size_t* held) noexcept {
    *span = guest_reserved.load(std::memory_order_relaxed);
    *held = guest_committed.load(std::memory_order_relaxed);
}

// Failure-only accounting uses stack storage; it must also work when allocation fails.
static void ReportSparseCommitFailure(const SparseRange& range, std::size_t slot) noexcept {
    const auto begin = range.begin.load(std::memory_order_relaxed);
    const auto span = range.end - begin;
    std::size_t owned = 0;
    for (std::size_t i = 0; i < span / SparseSlot; ++i)
        owned += range.owned[i].load(std::memory_order_relaxed);
    char line[384];
    std::snprintf(line, sizeof(line),
        "EDEN_SPARSE_COMMIT_FAILURE range=%p span=%zu slot=%zu slot_bytes=%zu guest=%u range_held=%zu tables_held=%zu tables_reserved=%zu guest_held=%zu guest_reserved=%zu\n",
        reinterpret_cast<void*>(begin), span, slot, SparseSlot, unsigned(range.guest), owned * SparseSlot,
        sparse_committed.load(std::memory_order_relaxed), sparse_reserved.load(std::memory_order_relaxed),
        guest_committed.load(std::memory_order_relaxed), guest_reserved.load(std::memory_order_relaxed));
    Report(line);
}

void CommitSparsePage(std::uintptr_t page) noexcept {
    SparseRange* range = SparseRangeOf(page);
    if (range == nullptr) return; // dense: already writable
    const auto begin = range->begin.load(std::memory_order_relaxed);
    const std::size_t slot = (page - begin) / SparseSlot;
    if (range->owned[slot].load(std::memory_order_acquire)) return;
    // Out of memory here has no way on (the caller writes the slot next), and it ended Ocarina of
    // Time on the console with a SIGABRT in MapPhysicalMemory (crash-20261110-234005). The pool
    // governor (src/direct_memory.c) lets the commit wait for memory instead: up to
    // EDEN_GOV_COMMIT_WAIT_MS (default 8000; 0: end at once, as before), asking the reclaimers while
    // the caches evict, in rounds of 250 ms with no lock of ours held.
#ifdef PS5_NATIVE
    const unsigned wait_ms = eden_governor_wait_ms ? eden_governor_wait_ms("EDEN_GOV_COMMIT_WAIT_MS", 8000) : 0;
#else
    const unsigned wait_ms = 0;
#endif
    const auto wait_for_memory = [&](unsigned& waited) {
#ifdef PS5_NATIVE
        if (waited >= wait_ms || !eden_governor_wait) return false;
        const unsigned round = std::min(250U, wait_ms - waited);
        waited += round;
        (void)eden_governor_wait(SparseSlot, round, range->guest ? 1 : GovernorTables);
        return true;
#else
        (void)waited;
        return false;
#endif
    };
    if (range->state != nullptr) {
        // A range with slot states commits under the slot's lock (guest_memory_service.inc).
        for (unsigned waited = 0; !ServiceCommit(*range, slot);) {
            if (wait_for_memory(waited)) continue;
            ReportSparseCommitFailure(*range, slot);
            Report("EDEN_SPARSE_TABLES commit failed: out of memory\n");
            std::abort();
        }
        return;
    }
    std::unique_lock lock{sparse_mutex};
    if (range->owned[slot].load(std::memory_order_relaxed)) return;
    for (unsigned waited = 0; !OwnSlot(begin + slot * SparseSlot, &range->physical[slot], range->guest);) {
        lock.unlock();
        const bool again = wait_for_memory(waited);
        lock.lock();
        if (range->owned[slot].load(std::memory_order_relaxed)) return; // another thread did it meanwhile
        if (again) continue;
        // Out of memory in the middle of a guest mapping: there is no table to continue with.
        ReportSparseCommitFailure(*range, slot);
        Report("EDEN_SPARSE_TABLES commit failed: out of memory\n");
        std::abort();
    }
    (range->guest ? guest_committed : sparse_committed).fetch_add(SparseSlot, std::memory_order_relaxed);
    range->owned[slot].store(true, std::memory_order_release);
}

// Eden gives back a table page once a guest unmapped all of it, and expects zeroes there
// afterwards. A slot's block is shared by every page of the slot, so the page is zeroed and the
// block stays: a later CommitSparsePage for it finds the slot owned. A page of a slot that was
// never written is the shared block of zeroes, read-only, and is left alone.
void DecommitVectorPage(std::uintptr_t page) noexcept {
    const long size = sysconf(_SC_PAGESIZE);
    if (size <= 0) std::abort();
    if (SparseRange* range = SparseRangeOf(page)) {
        const std::size_t slot = (page - range->begin.load(std::memory_order_relaxed)) / SparseSlot;
        if (!range->owned[slot].load(std::memory_order_acquire)) return;
    }
    std::memset(reinterpret_cast<void*>(page), 0, static_cast<std::size_t>(size));
}

#include "guest_memory_service.inc"

// Address space the sparse tables span, and the memory they hold (for the log).
void SparseUsage(std::size_t* reserved, std::size_t* committed) noexcept {
    *reserved = sparse_reserved.load(std::memory_order_relaxed);
    *committed = sparse_committed.load(std::memory_order_relaxed);
}

// ---- A range that grows: the heap ----
//
void* ReserveAddressRange(std::size_t size) noexcept;
bool CommitMemoryRangeFrom(void* address, std::size_t size, bool high) noexcept;
int DecommitMemoryRange(void* address, std::size_t size) noexcept;

// ReserveMemoryRange takes address space only. CommitMemoryRange backs a part of it with zeroed
// memory; DecommitMemoryRange gives that memory back to the system and keeps the address space.
// All work in whole large pages; the heap commits its range from the start, one piece after
// another, as allocations need it, and gives back pieces that hold nothing
// (headless/heap_arenas.inc).
//
// A range is committed in blocks of direct memory as large as the pool has room for, down to one
// large page: the range is one in address, its memory need not be. Asked for in one block, a
// 486 MiB request found no free block that large in a pool with a gigabyte free in pieces, and a
// game's session ended (a Tears of the Kingdom tester, v1.010.005). Each block is recorded, so the
// range can be given back block by block.
namespace {
struct CommittedBlock {
    std::uintptr_t address; // 0: a free entry
    std::size_t size;
    std::int64_t physical;
};
constexpr std::size_t MostCommittedBlocks = 8192;
CommittedBlock committed_blocks[MostCommittedBlocks];
// A spin lock, as the heap's own: these run inside malloc, where a mutex could allocate.
std::atomic_flag committed_lock = ATOMIC_FLAG_INIT;
void LockCommitted() noexcept {
    for (unsigned spins = 0; committed_lock.test_and_set(std::memory_order_acquire); ++spins) {
        if (spins < 64) sched_yield();
        else usleep(50);
    }
}
void UnlockCommitted() noexcept {
    committed_lock.clear(std::memory_order_release);
}
bool RecordCommitted(std::uintptr_t address, std::size_t size, std::int64_t physical) noexcept {
    LockCommitted();
    bool recorded = false;
    for (auto& block : committed_blocks) {
        if (block.address == 0) {
            block = CommittedBlock{address, size, physical};
            recorded = true;
            break;
        }
    }
    UnlockCommitted();
    return recorded;
}
std::atomic<std::size_t> committed_split{0}; // ranges that took more than one block
} // namespace

void* ReserveMemoryRange(std::size_t size) noexcept {
    if (size == 0 || size % LargePage != 0) return nullptr;
#ifdef PS5_NATIVE
    // Development A/B: dev-settings heap=whole takes the heap's memory at start, as before.
    if (DevSetting("heap=whole")) return nullptr;
#endif
    return ReserveAddressRange(size);
}

// Address space alone, whatever the heap's development switches say (the heap's large blocks).
void* ReserveAddressRange(std::size_t size) noexcept {
    if (size == 0 || size % LargePage != 0) return nullptr;
#ifdef PS5_NATIVE
    void* address = reinterpret_cast<void*>(cpu_mapping_hint);
    if (sceKernelReserveVirtualRange(&address, size, 0, LargePage) != 0) return nullptr;
    if (!cpu_mapping_range(address, size)) {
        (void)munmap(address, size);
        return nullptr;
    }
    return address;
#else
    void* address = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return address == MAP_FAILED ? nullptr : address;
#endif
}

bool CommitMemoryRange(void* address, std::size_t size) noexcept {
    return CommitMemoryRangeFrom(address, size, true);
}

// `high`: from the top of the pool (memory that stays: the heap's pieces), else from where the
// system places it (memory a session gives back: a large block).
bool CommitMemoryRangeFrom(void* address, std::size_t size, bool high) noexcept {
    if (address == nullptr || size == 0 || size % LargePage != 0 ||
        reinterpret_cast<std::uintptr_t>(address) % LargePage != 0)
        return false;
#ifdef PS5_NATIVE
    const auto begin = reinterpret_cast<std::uintptr_t>(address);
    std::size_t done = 0, block = size;
    unsigned blocks = 0, reserve_blocks = 0;
    while (done < size) {
        block = std::min(block, size - done);
        std::int64_t physical = -1;
        const int taken = DirectOwned(GovernorHeap, high, block, LargePage, &physical);
        if (taken != 0) {
            if (block != LargePage) {
                block = std::max(LargePage, block / 2 / LargePage * LargePage);
                continue;
            }
            // Only durable host heap pieces may use the emergency transfer.
            // Low-pool large allocations retain their existing failure policy.
            if (!high || !TakeHostReserveBlock(&physical)) break;
            ++reserve_blocks;
        }
        void* at = reinterpret_cast<void*>(begin + done);
        if (sceKernelMapDirectMemory(&at, block, PROT_READ | PROT_WRITE, MAP_FIXED, physical, LargePage) != 0 ||
            at != reinterpret_cast<void*>(begin + done)) {
            (void)sceKernelReleaseDirectMemory(physical, block);
            break;
        }
        if (!RecordCommitted(begin + done, block, physical)) {
            // No entry left to give it back with: unmap it now, keeping the address space.
            if (ReserveOver(at, block, LargePage) >= 0) (void)sceKernelReleaseDirectMemory(physical, block);
            break;
        }
        done += block;
        ++blocks;
    }
    if (done < size) {
        // What was taken goes back: the caller has no use for part of a range.
        if (done != 0) (void)DecommitMemoryRange(address, done);
        return false;
    }
    if (blocks > 1) committed_split.fetch_add(1, std::memory_order_relaxed);
    // The heap grows through here: no stdio (it may allocate).
    char line[160];
    const int length = std::snprintf(line, sizeof(line), "EDEN_HEAP_PIECE bytes=%zu va=%p blocks=%u%s reserve_blocks=%u\n", size,
                                     address, blocks, high ? "" : " large", reserve_blocks);
    if (length > 0) Note(line);
    std::memset(address, 0, size);
    return true;
#else
    (void)high;
    return mprotect(address, size, PROT_READ | PROT_WRITE) == 0;
#endif
}

// Gives back the memory of a committed part of a range, keeping its address space reserved (no
// other mapping can land there). 1: given back; 0: nothing was done (the range stays as it was);
// -1: given back in part (the range must not be used again). dev-settings heap_return=off keeps
// everything (0).
int DecommitMemoryRange(void* address, std::size_t size) noexcept {
    if (address == nullptr || size == 0) return 0;
#ifdef PS5_NATIVE
    static const bool kept = DevSetting("heap_return=off");
    if (kept) return 0;
    const auto begin = reinterpret_cast<std::uintptr_t>(address);
    const auto end = begin + size;
    int result = 1;
    std::size_t covered = 0;
    LockCommitted();
    for (auto& block : committed_blocks) {
        if (block.address < begin || block.address >= end || block.address == 0) continue;
        // As the guest's memory gives a block back (src/host_memory.cpp): the mapping becomes
        // reserved address space (ReserveOver), then the memory is released.
        const int reserved = ReserveOver(reinterpret_cast<void*>(block.address), block.size, LargePage);
        if (reserved < 0) {
            result = -1;
            continue;
        }
        if (reserved == 0) result = -1;
        if (sceKernelReleaseDirectMemory(block.physical, block.size) != 0) result = -1;
        covered += block.size;
        block = CommittedBlock{};
    }
    UnlockCommitted();
    if (covered != size) result = covered == 0 ? 0 : -1;
    return result;
#else
    void* at = mmap(address, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
    return at == address ? 1 : 0;
#endif
}

// A reserved range given back whole: its memory, then its address space.
void ReleaseMemoryRange(void* address, std::size_t size) noexcept {
    if (address == nullptr || size == 0) return;
#ifdef PS5_NATIVE
    // The release of a large block does not depend on the development switch.
    const auto begin = reinterpret_cast<std::uintptr_t>(address);
    const auto end = begin + size;
    LockCommitted();
    for (auto& block : committed_blocks) {
        if (block.address < begin || block.address >= end || block.address == 0) continue;
        if (munmap(reinterpret_cast<void*>(block.address), block.size) != 0 ||
            sceKernelReleaseDirectMemory(block.physical, block.size) != 0)
            std::abort();
        block = CommittedBlock{};
    }
    UnlockCommitted();
#endif
    (void)munmap(address, size);
}

// Ranges committed in more than one block of memory since the app started.
std::size_t CommittedSplit() noexcept {
    return committed_split.load(std::memory_order_relaxed);
}

// ---- A range that fills: a JIT code cache ----
//
// The JIT writes a code cache from its start, one block after another, and starts again at the
// beginning when the cache is full. Taken whole, the four caches of a 64-bit game are 656 MiB of
// the pool the CPU shares with the GPU, and a large game fills little more than half of that.
// So a cache is address space only, in two ranges (the one the JIT writes through and the one
// the code runs from), and memory is put behind both a step at a time as the JIT reaches it:
// CommitCodeRange is asked before every block. Memory is not taken back while the cache lives.
namespace {
constexpr std::size_t CodeStep = std::size_t{16} << 20;
constexpr std::size_t CodeSteps = 64; // the largest cache: 1 GiB
constexpr std::size_t CodePieces = CodeStep / LargePage; // a step's memory, in 2 MiB blocks
struct CodeRange {
    std::atomic<std::uintptr_t> executable{0}; // 0: free
    std::uintptr_t writable = 0;
    std::size_t span = 0;
    std::atomic<std::size_t> committed{0};
    std::size_t pieces = 0; // 2 MiB blocks mapped (a step stopped half way keeps those it has)
    std::int64_t physical[CodeSteps * CodePieces] = {}; // per 2 MiB block
};
std::array<CodeRange, 16> code_ranges;
std::mutex code_mutex;
std::atomic<std::size_t> code_reserved{0}, code_committed{0};
// Lock-free: asked in a fault handler.
bool InCodeCache(std::uintptr_t address) noexcept {
    for (const CodeRange& range : code_ranges) {
        const std::uintptr_t executable = range.executable.load(std::memory_order_acquire);
        if (executable != 0 && address >= executable && address - executable < range.span) return true;
    }
    return false;
}

// Memory behind the next step of both ranges, a 2 MiB block at a time: one block of 16 MiB
// asked of a pool cut into small free ranges was refused with 2 MiB blocks still free and the
// guest's reserve holding 512 MiB, and Mario Kart World stopped there. A block the pool has none
// for comes from that reserve, as a heap piece does (TakeHostReserveBlock leaves half of it for
// the guest's writes and gives way to writes that wait). code_mutex is held.
bool CommitCodeStep(CodeRange& range, std::size_t offset) {
    void* const writable = reinterpret_cast<void*>(range.writable + offset);
    void* const executable = reinterpret_cast<void*>(range.executable.load(std::memory_order_relaxed) + offset);
#ifdef PS5_NATIVE
    // A step that stopped half way goes on from the blocks it has (the JIT asks again).
    for (std::size_t piece = range.pieces - offset / LargePage; piece < CodePieces; ++piece) {
        std::int64_t physical = -1;
        if (DirectOwned(GovernorJit, false, LargePage, LargePage, &physical) != 0 && !TakeHostReserveBlock(&physical)) {
            // EDEN_JIT_RESERVE_LEND=1 (off unless asked for): a piece the pool refuses ends the
            // session ("the console has no memory for its next part": run 161, 20 MiB left while
            // the guest's reserve held 256 MiB at its floor; TakeHostReserveBlock lends only above
            // half the reserve, which a short pool never has). The reserve gives a block above its
            // floor back to the pool, as it does for a refused buffer, and the pool is asked once
            // more: a piece is one 2 MiB block, so any lent block serves. EDEN_GUEST_RESERVE_FLOOR
            // sets what may go.
            static const bool lend = std::getenv("EDEN_JIT_RESERVE_LEND") != nullptr && std::getenv("EDEN_JIT_RESERVE_LEND")[0] == '1';
            if (!lend || eden_guest_reserve_lend(LargePage) == 0 ||
                DirectOwned(GovernorJit, false, LargePage, LargePage, &physical) != 0)
                return false;
            std::fputs("EDEN_JIT_MEMORY a piece the pool refused came from the guest's reserve\n", stderr);
        }
        void* const piece_writable = static_cast<char*>(writable) + piece * LargePage;
        void* const piece_executable = static_cast<char*>(executable) + piece * LargePage;
        void* at = piece_writable;
        if (sceKernelMapDirectMemory(&at, LargePage, PROT_READ | PROT_WRITE, MAP_FIXED, physical, LargePage) != 0 ||
            at != piece_writable)
            return false;
        std::memset(piece_writable, 0, LargePage);
        at = piece_executable;
        if (sceKernelMapDirectMemory(&at, LargePage, PROT_READ, MAP_FIXED, physical, LargePage) != 0 ||
            at != piece_executable)
            return false;
        // As MapExecutableAlias: the mapping call refuses EXEC on some versions.
        if (mprotect(piece_executable, LargePage, PROT_READ | PROT_EXEC) != 0) return false;
        range.physical[offset / LargePage + piece] = physical;
        range.pieces = offset / LargePage + piece + 1;
    }
#else
    if (mprotect(writable, CodeStep, PROT_READ | PROT_WRITE) != 0 ||
        mprotect(executable, CodeStep, PROT_READ | PROT_EXEC) != 0)
        return false;
#endif
    code_committed.fetch_add(CodeStep, std::memory_order_relaxed);
    return true;
}
} // namespace

// Address space for a code cache of `size` bytes: returns the range to write through and sets
// the one to run from. Null when a cache this small is better taken whole, or when the
// platform refuses: the caller then takes the cache whole, as before.
void* ReserveCodeRange(std::size_t size, void** executable) noexcept {
    if (size < 2 * CodeStep || size % CodeStep != 0 || size / CodeStep > CodeSteps) return nullptr;
    const std::lock_guard lock{code_mutex};
    CodeRange* free_range = nullptr;
    for (auto& range : code_ranges)
        if (range.executable.load(std::memory_order_relaxed) == 0) {
            free_range = &range;
            break;
        }
    if (free_range == nullptr) return nullptr;
    void* views[2] = {nullptr, nullptr};
#ifdef PS5_NATIVE
    // Development A/B: dev-settings jit_memory=whole takes every cache whole, as before.
    if (DevSetting("jit_memory=whole")) return nullptr;
    (void)sceKernelEnableDmemAliasing(); // reports zero regardless; the first step is the check
    for (void*& view : views) {
        view = reinterpret_cast<void*>(cpu_mapping_hint);
        if (sceKernelReserveVirtualRange(&view, size, 0, LargePage) != 0) {
            view = nullptr;
        } else if (!cpu_mapping_range(view, size)) {
            (void)munmap(view, size);
            view = nullptr;
        }
    }
#else
    const int fd = memfd_create("eden-jit", MFD_CLOEXEC);
    if (fd >= 0 && ftruncate(fd, static_cast<off_t>(size)) == 0) {
        for (void*& view : views) {
            view = mmap(nullptr, size, PROT_NONE, MAP_SHARED, fd, 0);
            if (view == MAP_FAILED) view = nullptr;
        }
    }
    if (fd >= 0 && close(fd) != 0) std::abort();
#endif
    if (views[0] == nullptr || views[1] == nullptr) {
        for (void* view : views)
            if (view != nullptr) (void)munmap(view, size);
        return nullptr;
    }
    free_range->writable = reinterpret_cast<std::uintptr_t>(views[0]);
    free_range->span = size;
    free_range->committed.store(0, std::memory_order_relaxed);
    free_range->pieces = 0;
    free_range->executable.store(reinterpret_cast<std::uintptr_t>(views[1]), std::memory_order_release);
    code_reserved.fetch_add(size, std::memory_order_relaxed);
    *executable = views[1];
    return views[0];
}

// Memory behind the first `bytes` of the cache that runs from `executable` (more than the
// cache: all of it). True when it is there, and for a cache that was taken whole. False when
// the console has none left.
bool CommitCodeRange(const void* executable, std::size_t bytes) noexcept {
    const auto address = reinterpret_cast<std::uintptr_t>(executable);
    for (auto& range : code_ranges) {
        if (address == 0 || range.executable.load(std::memory_order_acquire) != address) continue;
        if (bytes > range.span) bytes = range.span;
        if (bytes <= range.committed.load(std::memory_order_acquire)) return true;
        const std::lock_guard lock{code_mutex};
        std::size_t committed = range.committed.load(std::memory_order_relaxed);
        while (committed < bytes) {
            if (!CommitCodeStep(range, committed)) return false;
            committed += CodeStep;
            range.committed.store(committed, std::memory_order_release);
        }
        char line[112];
        const int length = std::snprintf(line, sizeof(line), "EDEN_JIT_MEMORY code=%p held=%zu of=%zu\n", executable,
                                         committed, range.span);
        if (length > 0) Report(line);
        return true;
    }
    return true;
}

// Gives back a cache ReserveCodeRange made: its memory and both ranges. False for any other.
bool FreeCodeRange(void* executable) noexcept {
    const auto address = reinterpret_cast<std::uintptr_t>(executable);
    if (address == 0) return false;
    const std::lock_guard lock{code_mutex};
    for (auto& range : code_ranges) {
        if (range.executable.load(std::memory_order_relaxed) != address) continue;
        const std::size_t committed = range.committed.load(std::memory_order_relaxed);
        if (munmap(reinterpret_cast<void*>(range.writable), range.span) != 0 || munmap(executable, range.span) != 0)
            std::abort();
#ifdef PS5_NATIVE
        for (std::size_t piece = 0; piece < range.pieces; ++piece)
            if (sceKernelReleaseDirectMemory(range.physical[piece], LargePage) != 0) std::abort();
        range.pieces = 0;
#endif
        code_reserved.fetch_sub(range.span, std::memory_order_relaxed);
        code_committed.fetch_sub(committed, std::memory_order_relaxed);
        range.committed.store(0, std::memory_order_relaxed);
        range.executable.store(0, std::memory_order_release);
        return true;
    }
    return false;
}

// Address space the growing code caches span, and the memory behind them (for the log).
void CodeRangeUsage(std::size_t* reserved, std::size_t* committed) noexcept {
    *reserved = code_reserved.load(std::memory_order_relaxed);
    *committed = code_committed.load(std::memory_order_relaxed);
}
} // namespace Common
