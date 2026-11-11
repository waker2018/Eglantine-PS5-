"""Adapt the pinned Eden Vulkan platform boundary and required renderer fixes."""
from pathlib import Path
import struct
import subprocess
import sys
# --report: name every anchor that no longer matches (make update), instead of stopping at the first.
report = '--report' in sys.argv
source, output = map(Path, [a for a in sys.argv[1:] if a != '--report'])
port = Path(__file__).resolve().parents[1] / 'headless'
broken = []
def adapt(relative, destination, replacements):
    text = (source / relative).read_text()
    for old, new in replacements:
        if relative.endswith('/vk_rasterizer.cpp') and old.endswith('// update engine as channel may be different.') and old not in text:
            marker = '// Donkey Kong Bananza\'s terrain:'
            if marker in text:
                old = old.replace('// update engine as channel may be different.', marker)
                new = new.replace('// update engine as channel may be different.', marker)
        if relative.endswith('/vk_rasterizer.cpp') and old.startswith('    pipeline->SetEngine(maxwell3d, gpu_memory);') and old not in text:
            engine = '    pipeline->SetEngine(maxwell3d, gpu_memory);\n'
            configure = '    if (Profile::Scope scope{Profile::Zone::Configure}; !pipeline->Configure(is_indexed)) {'
            begin = text.index(engine) + len(engine)
            end = text.index(configure, begin)
            repack = text[begin:end]
            old = old.replace(engine, engine + repack)
            new = new.replace(configure, repack + configure)
        if relative.endswith('/vulkan_memory_allocator.cpp') and old.startswith('    vk::Check(vmaCreateBuffer(') and old not in text:
            if 'VkResult result = vmaCreateBuffer(' in text:
                # This core already retries outside the driver budget on allocation failure.
                if 'host_ci.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;' in text:
                    continue
        if text.count(old) != 1:
            if report:
                broken.append(f"{relative}: {old[:70]!r} (matches {text.count(old)})")
                continue
            # This tree follows a newer core than the anchors were written for: an edit that no
            # longer fits is left out and listed in the build log, not a stop.
            print(f"[ANCHOR-SKIPPED] {relative}: {old[:90]!r} (matches {text.count(old)})")
            continue
        text = text.replace(old, new)
    path = output / destination
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)

adapt('src/core/frontend/emu_window.h', 'include/core/frontend/emu_window.h',
      [('    Xcb,', '    Xcb,\n    PS5,')])
adapt('src/video_core/renderer_vulkan/vk_rasterizer.cpp', 'vulkan_rasterizer.cpp', [
    # Compute synchronization (dev_vulkan.h compute_barriers): indirect
    # dispatches returned before the pre-dispatch barrier, and no barrier made a dispatch's
    # writes visible to later indirect arguments, vertex/index fetch, shaders or copies.
    ('void RasterizerVulkan::DispatchCompute() {',
     """namespace {
constexpr VkMemoryBarrier COMPUTE_WRITE_BARRIER{
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .pNext = nullptr,
    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_INDEX_READ_BIT |
                     VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_UNIFORM_READ_BIT |
                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                     VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
};
constexpr VkMemoryBarrier COMPUTE_INPUT_BARRIER{
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .pNext = nullptr,
    .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
};
constexpr VkPipelineStageFlags COMPUTE_CONSUMER_STAGES =
    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
    VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
    VK_PIPELINE_STAGE_TRANSFER_BIT;
} // namespace

void RasterizerVulkan::DispatchCompute() {"""),
    # Anchored on the dispatch record itself: 0347-up-compute-indirect-order puts its own
    # (switched) barrier between RequestOutsideRenderPassOperationContext and this record, and
    # ours stays right before the dispatch, after it.
    ('        scheduler.Record([pipeline, indirect_buffer = buffer->Handle(),',
     """        if (::Eden::DevVulkan::compute_barriers) {
            scheduler.Record([](vk::CommandBuffer cmdbuf) {
                cmdbuf.PipelineBarrier(vk::PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER,
                                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                       0, COMPUTE_INPUT_BARRIER);
            });
        }
        scheduler.Record([pipeline, indirect_buffer = buffer->Handle(),"""),
    ('            cmdbuf.DispatchIndirect(indirect_buffer, indirect_offset);\n',
     """            cmdbuf.DispatchIndirect(indirect_buffer, indirect_offset);
            if (::Eden::DevVulkan::compute_barriers)
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, COMPUTE_CONSUMER_STAGES, 0,
                                       COMPUTE_WRITE_BARRIER);
"""),
    ('        cmdbuf.Dispatch(dim[0], dim[1], dim[2]);\n',
     """        cmdbuf.Dispatch(dim[0], dim[1], dim[2]);
        if (::Eden::DevVulkan::compute_barriers)
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, COMPUTE_CONSUMER_STAGES, 0,
                                   COMPUTE_WRITE_BARRIER);
"""),
    # CounterEnable below starts a query immediately. A deferred clear has no
    # active pass yet, so later realization would end that query in a different
    # render-pass scope (VUID-vkCmdEndQuery-None-07007). Keep the immediate path
    # when the guest enables occlusion counting.
    ('const bool can_defer_clear = ENABLE_DEFERRED_CLEAR &&',
     'const bool can_defer_clear = ENABLE_DEFERRED_CLEAR && !regs.zpass_pixel_count_enable &&'),
    ('#include <algorithm>', '#include <algorithm>\n#include <thread>\n#include "performance.h"'),
    # A refused allocation (MemoryAllocator::ReleaseRefused, ReportRefused) reaches the caches here.
    # It can come from inside them, so their locks are only tried, and the staging pool, which has no
    # lock, is touched only on the thread that runs it: the one that ticks it.
    ('RasterizerVulkan::RasterizerVulkan(Core::Frontend::EmuWindow& emu_window_, Tegra::GPU& gpu_,',
     'namespace {\n'
     'std::atomic<std::thread::id> rasterizer_thread{};\n'
     '}\n\n'
     'RasterizerVulkan::RasterizerVulkan(Core::Frontend::EmuWindow& emu_window_, Tegra::GPU& gpu_,'),
    ('    scheduler.SetQueryCache(query_cache);\n',
     '''    scheduler.SetQueryCache(query_cache);
    memory_allocator.SetRefusalHook([](void* argument, bool release) -> u64 {
        auto* const self = static_cast<RasterizerVulkan*>(argument);
        const bool owner = std::this_thread::get_id() == rasterizer_thread.load(std::memory_order_relaxed);
        if (release) {
            return owner ? self->staging_pool.ReleaseCompleted() : 0;
        }
        if (owner) {
            self->staging_pool.ReportHeld();
        } else {
            std::fprintf(stderr, "EDEN_REFUSED_STAGING not read: refused on another thread\\n");
        }
        if (std::unique_lock lock{self->texture_cache.mutex, std::try_to_lock}; lock) {
            self->texture_cache.ReportMemory();
        } else {
            std::fprintf(stderr, "EDEN_REFUSED_IMAGES not read: another thread holds the cache\\n");
        }
        if (std::unique_lock lock{self->buffer_cache.mutex, std::try_to_lock}; lock) {
            self->buffer_cache.ReportMemory();
        } else {
            std::fprintf(stderr, "EDEN_REFUSED_BUFFERS not read: another thread holds the cache\\n");
        }
        return 0;
    }, this);
'''),
    ('RasterizerVulkan::~RasterizerVulkan() {\n',
     'RasterizerVulkan::~RasterizerVulkan() {\n    memory_allocator.SetRefusalHook(nullptr, nullptr);\n'),
    ('    staging_pool.TickFrame();\n',
     '    rasterizer_thread.store(std::this_thread::get_id(), std::memory_order_relaxed);\n'
     '    staging_pool.TickFrame();\n'),
    # Guest threads take the cache locks here (tracked-page writes, flush-area reads).
    ('bool RasterizerVulkan::OnCPUWrite(DAddr addr, u64 size) {',
     'bool RasterizerVulkan::OnCPUWrite(DAddr addr, u64 size) {\n    ::Eden::Performance::Timer guest_write_timer(::Eden::Performance::guest_cpu_write);'),
    ('VideoCore::RasterizerDownloadArea RasterizerVulkan::GetFlushArea(DAddr addr, u64 size) {',
     'VideoCore::RasterizerDownloadArea RasterizerVulkan::GetFlushArea(DAddr addr, u64 size) {\n    ::Eden::Performance::Timer guest_read_timer(::Eden::Performance::guest_cpu_read);'),
    # ... and wait for those locks spinning briefly before sleeping (performance.h GuestCacheLock).
    # 0316 (EDEN_PE_CPU_READ_SKIP) turned this lock into a deferred unique_lock; its blocking path keeps the spin.
    ('        if (!skip) {\n            lock.lock();\n',
     '        if (!skip) {\n            ::Eden::Performance::GuestCacheLock(texture_cache.mutex);\n'
     '            lock = std::unique_lock{texture_cache.mutex, std::adopt_lock};\n'),
    ('        std::scoped_lock lock{buffer_cache.mutex};\n        if (buffer_cache.OnCPUWrite(addr, size)) {',
     '        ::Eden::Performance::GuestCacheLock(buffer_cache.mutex);\n'
     '        std::lock_guard lock{buffer_cache.mutex, std::adopt_lock};\n        if (buffer_cache.OnCPUWrite(addr, size)) {'),
    ('        std::scoped_lock lock{texture_cache.mutex};\n        texture_cache.WriteMemory(addr, size);\n    }\n    pipeline_cache.InvalidateRegion(addr, size);',
     '        ::Eden::Performance::GuestCacheLock(texture_cache.mutex);\n'
     '        std::lock_guard lock{texture_cache.mutex, std::adopt_lock};\n        texture_cache.WriteMemory(addr, size);\n    }\n    pipeline_cache.InvalidateRegion(addr, size);'),
    # The upstream code has an __ANDROID__ ifdef for DRAWS_TO_DISPATCH (512 vs 4096) and
    # CHECK_MASK (3 vs 7). Neither branch applies: replace the whole block. 512 draws per
    # command buffer submission: 4,096 exceeds the PS5 GPU watchdog timeout in shader-heavy
    # scenes (VK_ERROR_DEVICE_LOST in Xenoblade 2's cities). CHECK_MASK is the configurable
    # dispatch interval (dev-settings dispatch_draws=N); default 64 draws between hand-offs.
    ('#ifdef __ANDROID__\n    static constexpr u32 DRAWS_TO_DISPATCH = 512;\n    static constexpr u32 CHECK_MASK = 3;\n#else\n    // Every 32 draws (was 8): each dispatch woke the worker thread, and Switch 2 titles make\n    // ~55,000 draws a second. A full chunk is dispatched on its own (Scheduler::Record).\n    static constexpr u32 DRAWS_TO_DISPATCH = 4096;\n    static constexpr u32 CHECK_MASK = 31;\n#endif // __ANDROID__\n\n    static_assert(DRAWS_TO_DISPATCH % (CHECK_MASK + 1) == 0);\n',
     '    static constexpr u32 DRAWS_TO_DISPATCH = 512;\n    const u32 CHECK_MASK = ::Eden::Performance::dispatch_mask.load(std::memory_order_relaxed);\n\n'),
    # Where the ~4 us a draw spends goes: six phases under one running clock, each Mark taking the
    # gap since the previous one so they do not overlap. The Timer still wraps the whole function,
    # so a draw that returns early is charged for its unmarked tail and the sum of the six falls
    # short by that amount plus two clock reads per phase. control/scheduler.cpp aside, this is the
    # only place the dispatch's time can be spent - so its six numbers should add up to it.
    ('''    namespace Profile = VideoCommon::Nx2Profile;
    {
        Profile::Scope scope{Profile::Zone::FlushWork};
        FlushWork();
        gpu_memory->FlushCaching();
    }
''',
     '''    namespace Profile = VideoCommon::Nx2Profile;
    const ::Eden::Performance::Timer rasterizer_draw_timer(::Eden::Performance::rasterizer_draw);
    auto phase = ::Eden::Performance::MarkStart();
    {
        Profile::Scope scope{Profile::Zone::FlushWork};
        FlushWork();
        gpu_memory->FlushCaching();
    }
    ::Eden::Performance::Mark(::Eden::Performance::draw_flush, phase);
'''),
    ('''        return pipeline_cache.CurrentGraphicsPipeline();
    }()};
''',
     '''        return pipeline_cache.CurrentGraphicsPipeline();
    }()};
    ::Eden::Performance::Mark(::Eden::Performance::draw_pipeline, phase);
'''),
    ('''    if (Profile::Scope scope{Profile::Zone::Configure}; !pipeline->Configure(is_indexed)) {
        report_failure("configure failed");
        return;
    }
''',
     '''    // Waiting for both cache mutexes. SetEngine and the vertex repack layout between them and here
    // are cheap, and the hold itself carries on through what follows: the phases below measure it.
    ::Eden::Performance::Mark(::Eden::Performance::draw_lock_wait, phase);
    if (Profile::Scope scope{Profile::Zone::Configure}; !pipeline->Configure(is_indexed)) {
        report_failure("configure failed");
        return;
    }
    ::Eden::Performance::Mark(::Eden::Performance::draw_config, phase);
'''),
    ('''    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, maxwell3d->regs.zpass_pixel_count_enable);
    {
        Profile::Scope scope{Profile::Zone::DrawRecord};
        draw_func();
    }
''',
     '''    query_cache.CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, maxwell3d->regs.zpass_pixel_count_enable);
    ::Eden::Performance::Mark(::Eden::Performance::draw_state, phase);
    {
        Profile::Scope scope{Profile::Zone::DrawRecord};
        draw_func();
    }
    ::Eden::Performance::Mark(::Eden::Performance::draw_record, phase);
'''),
    # A forced release blocks the GPU thread until the fence thread has retired
    # every older fence. Time only that producer-side drain.
    ('    fence_manager.WaitPendingFences(force);',
     '''    if (force) {
        ::Eden::Performance::Timer drain_timer(::Eden::Performance::gpu_fence_drain);
        fence_manager.WaitPendingFences(force);
    } else {
        fence_manager.WaitPendingFences(force);
    }'''),
    # UpdateDynamicStates asked the cache for the current pipeline a second time
    # per draw, recomputing the same fixed-state key. Reuse PrepareDraw's result;
    # other callers keep the original lookup.
    ('template <typename Func>\nvoid RasterizerVulkan::PrepareDraw(bool is_indexed, Func&& draw_func) {',
     '''namespace {
// GPU thread only: the pipeline PrepareDraw refreshed for the draw in progress.
GraphicsPipeline* prepared_pipeline = nullptr;
} // namespace

template <typename Func>
void RasterizerVulkan::PrepareDraw(bool is_indexed, Func&& draw_func) {'''),
    ('    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};\n    // update engine as channel may be different.',
     '''    std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
    prepared_pipeline = pipeline;
    SCOPE_EXIT {
        prepared_pipeline = nullptr;
    };
    // update engine as channel may be different.'''),
    # 0400 (EDEN_GBB_PIPELINE_ONCE) passes draw_pipeline; without it, keep using the prepared draw's pipeline.
    ('            draw_pipeline != nullptr ? draw_pipeline : pipeline_cache.CurrentGraphicsPipeline();',
     '            draw_pipeline != nullptr ? draw_pipeline\n            : prepared_pipeline   ? prepared_pipeline\n                                  : pipeline_cache.CurrentGraphicsPipeline();'),
    # Development GPU fault probe (dev_vulkan.h, pipeline_trace=on): once the runner
    # creates /app0/sync-draws.txt, report every draw/dispatch to klog and wait for the
    # GPU, so the last report before a fatal GPU fault names the operation.
    ('#include "performance.h"', '#include "performance.h"\n#include "dev_vulkan.h"\n#include "diagnostics.h"\n#include <filesystem>'),
    ('GraphicsPipeline* prepared_pipeline = nullptr;',
     'GraphicsPipeline* prepared_pipeline = nullptr;\n'
     '// The pipeline of the last prepared draw, for the development fault probe.\n'
     'GraphicsPipeline* probe_pipeline = nullptr;\n'
     'bool SyncProbeActive() {\n'
     '    static unsigned poll = 0;\n'
     '    static bool active = false;\n'
     '    if (!active && ::Eden::DevVulkan::trace_pipelines && (++poll & 255) == 0)\n'
     '        active = std::filesystem::exists("/app0/sync-draws.txt");\n'
     '    return active;\n'
     '}\n'
     'void SyncProbe(Scheduler& scheduler, const char* kind, const void* pipeline) {\n'
     '    if (!SyncProbeActive()) return;\n'
     '    static unsigned long long sequence = 0;\n'
     '    char text[96];\n'
     '    std::snprintf(text, sizeof(text), "%s #%llu p=%p", kind, ++sequence, pipeline);\n'
     '    ::Eden::Report("sync", text);\n'
     '    scheduler.Finish();\n'
     '}'),
    ('    prepared_pipeline = pipeline;\n', '    prepared_pipeline = pipeline;\n    probe_pipeline = pipeline;\n'),
    ('        }\n    });\n}\n\nvoid RasterizerVulkan::DrawIndirect() {',
     '        }\n    });\n    SyncProbe(scheduler, "draw", probe_pipeline);\n}\n\nvoid RasterizerVulkan::DrawIndirect() {'),
    ('void RasterizerVulkan::DrawTexture() {', 'void RasterizerVulkan::DrawTexture() {\n    SCOPE_EXIT { SyncProbe(scheduler, "draw_texture", nullptr); };'),
    ('void RasterizerVulkan::Clear(u32 layer_count) {', 'void RasterizerVulkan::Clear(u32 layer_count) {\n    SCOPE_EXIT { SyncProbe(scheduler, "clear", nullptr); };'),
    ('    ComputePipeline* const pipeline{pipeline_cache.CurrentComputePipeline()};\n    if (!pipeline) {\n        if (kepler_compute->IsNativeMode()) throw Shader::RuntimeError("Native compute pipeline unavailable");\n        return;\n    }',
     '    ComputePipeline* const pipeline{pipeline_cache.CurrentComputePipeline()};\n    if (!pipeline) {\n        if (kepler_compute->IsNativeMode()) throw Shader::RuntimeError("Native compute pipeline unavailable");\n        return;\n    }\n'
     '    SCOPE_EXIT { SyncProbe(scheduler, "dispatch", pipeline); };'),
])
# Host conditional rendering (VK_EXT_conditional_rendering, dev_vulkan.h hcr_mode): count each
# decision (EDEN_DEV_GUEST cond= slots 13-19); hcr=exact sends the conditions Eden would draw
# unconditionally, or predicate from a pending host query, to the exact CPU evaluation;
# hcr=cpu evaluates every condition on the CPU and predicates from constant 0/1 buffers.
adapt('src/video_core/renderer_vulkan/vk_query_cache.cpp', 'vulkan_query_cache.cpp', [
    ('#include "video_core/vulkan_common/vulkan_wrapper.h"\n',
     '#include "video_core/vulkan_common/vulkan_wrapper.h"\n#include "dev_vulkan.h"\n#include "performance.h"\n'),
    ('    vk::Buffer hcr_resolve_buffer;\n',
     '    vk::Buffer hcr_resolve_buffer;\n    vk::Buffer hcr_constant_buffer; // hcr=cpu: predicates 0 and 1\n'),
    ('        hcr_resolve_buffer = memory_allocator.CreateBuffer(buffer_ci, MemoryUsage::DeviceLocal);\n    }\n',
     '''        hcr_resolve_buffer = memory_allocator.CreateBuffer(buffer_ci, MemoryUsage::DeviceLocal);
        if (device.IsExtConditionalRendering() && ::Eden::DevVulkan::hcr_mode == 2) {
            VkBufferCreateInfo constant_ci = buffer_ci;
            constant_ci.size = 2 * sizeof(u32);
            constant_ci.usage = VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT;
            hcr_constant_buffer = memory_allocator.CreateBuffer(constant_ci, MemoryUsage::Upload);
            const u32 predicates[2]{0, 1};
            std::memcpy(hcr_constant_buffer.Mapped().data(), predicates, sizeof(predicates));
            hcr_constant_buffer.Flush();
        }
    }
'''),
    ('''bool QueryCacheRuntime::HostConditionalRenderingCompareValue(VideoCommon::LookupData object_1,
                                                             [[maybe_unused]] bool qc_dirty) {
    if (!impl->device.IsExtConditionalRendering()) {
        return false;
    }
    HostConditionalRenderingCompareBCImpl(object_1.address, true, true);
    return true;
}''',
     '''namespace {
// hcr=cpu: Maxwell3D::ProcessQueryCondition's exact evaluation (the read flushes pending GPU
// results), then the driver predicates the draws from a constant buffer. kind: 0 Conditional,
// 1 IfEqual, 2 IfNotEqual.
bool HostConditionalRenderingOnCpu(QueryCacheRuntime& runtime, QueryCacheRuntimeImpl& impl, DAddr address,
                                   unsigned kind) {
    if (address == 0 || !impl.hcr_constant_buffer) {
        ::Eden::Performance::CountCondition(13);
        runtime.EndHostConditionalRendering();
        return false;
    }
    u32 compare[6]{};
    impl.device_memory.ReadBlock(address, compare, sizeof(compare));
    const bool equal = compare[0] == compare[4] && compare[1] == compare[5];
    const bool execute = kind == 0 ? compare[0] != 0 && compare[1] != 0 : (kind == 1 ? equal : !equal);
    ::Eden::Performance::CountCondition(execute ? 17 : 18);
    const bool was_running = impl.is_hcr_running;
    if (was_running) {
        runtime.PauseHostConditionalRendering();
    }
    impl.hcr_setup.buffer = *impl.hcr_constant_buffer;
    impl.hcr_setup.offset = execute ? sizeof(u32) : 0;
    impl.hcr_setup.flags = 0;
    impl.hcr_is_set = true;
    impl.is_hcr_running = false;
    if (was_running) {
        runtime.ResumeHostConditionalRendering();
    }
    return true;
}
} // namespace

bool QueryCacheRuntime::HostConditionalRenderingCompareValue(VideoCommon::LookupData object_1,
                                                             [[maybe_unused]] bool qc_dirty) {
    if (!impl->device.IsExtConditionalRendering()) {
        return false;
    }
    if (::Eden::DevVulkan::hcr_mode == 2) {
        return HostConditionalRenderingOnCpu(*this, *impl, object_1.address, 0);
    }
    if (::Eden::DevVulkan::hcr_mode == 1 && qc_dirty) {
        ::Eden::Performance::CountCondition(19);
        EndHostConditionalRendering();
        return false;
    }
    ::Eden::Performance::CountCondition(16);
    HostConditionalRenderingCompareBCImpl(object_1.address, true, true);
    return true;
}'''),
    ('''    if (!impl->device.IsExtConditionalRendering()) {
        return false;
    }

    const auto check_in_bc = [&](DAddr address) {''',
     '''    if (!impl->device.IsExtConditionalRendering()) {
        return false;
    }
    if (::Eden::DevVulkan::hcr_mode == 2) {
        return HostConditionalRenderingOnCpu(*this, *impl, object_1.address, equal_check ? 1 : 2);
    }

    const auto check_in_bc = [&](DAddr address) {'''),
    ('''    if (!is_in_ac[0] && !is_in_ac[1]) {
        EndHostConditionalRendering();
        return false;
    }

    if (!qc_dirty && !is_in_bc[0] && !is_in_bc[1]) {
        EndHostConditionalRendering();
        return false;
    }
''',
     '''    if (!is_in_ac[0] && !is_in_ac[1]) {
        ::Eden::Performance::CountCondition(13);
        EndHostConditionalRendering();
        return false;
    }

    if (!qc_dirty && !is_in_bc[0] && !is_in_bc[1]) {
        ::Eden::Performance::CountCondition(13);
        EndHostConditionalRendering();
        return false;
    }
    // hcr=exact: a pending host query is not in any buffer yet; the CPU evaluation waits for it.
    if (::Eden::DevVulkan::hcr_mode == 1 && qc_dirty) {
        ::Eden::Performance::CountCondition(19);
        EndHostConditionalRendering();
        return false;
    }
'''),
    ('''            HostConditionalRenderingCompareValueImpl(*objects[j], equal_check);
            return true;''',
     '''            ::Eden::Performance::CountCondition(15);
            HostConditionalRenderingCompareValueImpl(*objects[j], equal_check);
            return true;'''),
    ('''    if (!is_gpu_high) {
        EndHostConditionalRendering();
        return true;
    }

    if (!is_in_bc[0] && !is_in_bc[1]) {
        EndHostConditionalRendering();
        return true;
    }
    HostConditionalRenderingCompareBCImpl(object_1.address, equal_check);''',
     '''    // Eden draws these unconditionally; hcr=exact evaluates them on the CPU instead.
    if (!is_gpu_high || (!is_in_bc[0] && !is_in_bc[1])) {
        ::Eden::Performance::CountCondition(14);
        EndHostConditionalRendering();
        return ::Eden::DevVulkan::hcr_mode != 1;
    }
    ::Eden::Performance::CountCondition(16);
    HostConditionalRenderingCompareBCImpl(object_1.address, equal_check);'''),
])
adapt('src/video_core/renderer_vulkan/vk_swapchain.cpp', 'vulkan_swapchain.cpp', [
    # An acquisition error has no usable image index. Forward it to the existing
    # renderer recovery path instead of indexing or presenting an invalid image.
    ('LOG_ERROR(Render_Vulkan, "vkAcquireNextImageKHR returned {}", string_VkResult(result));',
     'LOG_ERROR(Render_Vulkan, "vkAcquireNextImageKHR returned {}", string_VkResult(result));\n        vk::Check(result);'),
    ('LOG_CRITICAL(Render_Vulkan, "Failed to present with error {}", string_VkResult(result));',
     'LOG_CRITICAL(Render_Vulkan, "Failed to present with error {}", string_VkResult(result));\n        vk::Check(result);'),
])
# Keep the overlay in the existing display composition pass (also used for
# frontend screenshots), excluding guest applet captures. Cache its pipeline.
# Preserve upstream ABI, including consumers whose quoted includes resolve to
# the original header. Display/capture purpose belongs to the draw call.
adapt('src/video_core/present.h', 'include/video_core/present.h', [])
adapt('src/video_core/renderer_vulkan/vk_present_manager.h',
      'include/video_core/renderer_vulkan/vk_present_manager.h', [
    ('#include <mutex>', '#include <mutex>\n#include <exception>'),
    ('    std::jthread present_thread;',
     '    std::exception_ptr present_failure;\n    std::jthread present_thread;'),
])
adapt('src/video_core/renderer_vulkan/vk_present_manager.cpp', 'vulkan_present_manager.cpp', [
    ('#include "video_core/renderer_vulkan/vk_present_manager.h"',
     '#include "video_core/renderer_vulkan/vk_present_manager.h"\n#include "gpu_failure.h"\n#include "performance.h"'),
    # GetRenderFrame runs on the GPU thread: time its free-frame and
    # present-completion waits (FIFO back-pressure on the producer).
    ('    // Wait for free presentation frames\n    std::unique_lock lock{free_mutex};',
     '    // Wait for free presentation frames\n    ::Eden::Performance::Timer present_wait_timer(::Eden::Performance::gpu_present_wait);\n    std::unique_lock lock{free_mutex};'),
    ('PresentThread(token);', '''try {
            PresentThread(token);
        } catch (...) {
            const auto error = std::current_exception();
            {
                std::scoped_lock lock{queue_mutex, free_mutex};
                present_failure = error;
                present_queue.clear();
            }
            free_cv.notify_all();
            frame_cv.notify_all();
            Eden::RecordGpuFailure(error);
        }'''),
    ('free_cv.wait(lock, [this] { return !free_queue.empty(); });',
     '''free_cv.wait(lock, [this] { return present_failure || !free_queue.empty(); });
    if (present_failure) std::rethrow_exception(present_failure);'''),
    ('            present_queue.push_back(frame);',
     '            if (present_failure) return;\n            present_queue.push_back(frame);'),
    ('frame_cv.wait(queue_lock, [this] { return present_queue.empty(); });',
     'frame_cv.wait(queue_lock, [this] { return present_failure || present_queue.empty(); });'),
])
# Descriptor buffers: Eden's null buffer (nullDescriptor devices) has device address 0,
# but WriteDescriptorBuffer passed it with the binding's range. RADV then builds a real
# descriptor at GPU address 0; desktop kernels only log the resulting page fault, the PS5
# kills the title (TCP read at VA 0).
# The spec requires a null pAddressInfo (or VK_WHOLE_SIZE) for a null descriptor.
# The allocator's budget is the driver's, and on the console it does not say what is available: the
# driver counts each allocation twice (as VRAM and as visible VRAM), so the budget of the 8 GiB heap
# is used up at about 4 GiB, and it knows nothing of the pool the CPU shares. Eden asks for images
# and buffers within the budget and takes a refusal as final. A refused image or buffer is asked for
# again outside the budget. When the driver refuses that too, the staging the GPU is done with and the
# heap's empty spaces go back and it is asked for once more; refused then, the session ends cleanly
# (a buffer has no fallback), and stderr.log says what held the pool (MemoryAllocator::ReportRefused).
ALLOCATOR_REFUSED = """    VkResult created = CALL;
    if (created != VK_SUCCESS) {
        VmaAllocationCreateInfo outside_budget = alloc_ci;
        outside_budget.flags &= ~static_cast<VmaAllocationCreateFlags>(VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT);
        created = AGAIN;
        static std::atomic<unsigned long long> refused{0};
        const auto times = refused.fetch_add(1, std::memory_order_relaxed) + 1;
        if ((times & (times - 1)) == 0) { // the 1st, 2nd, 4th, 8th ... time
            std::fprintf(stderr, "EDEN_GRAPHICS_MEMORY budget refused %llu requests; asked again outside it: result %d, DETAIL\\n",
                         times, int(created), VALUES);
        }
        if (created != VK_SUCCESS) {
            VmaBudget held[VK_MAX_MEMORY_HEAPS]{};
            vmaGetHeapBudgets(allocator, held);
            unsigned long long blocks = 0, used = 0;
            for (const VmaBudget& heap : held) {
                blocks += heap.statistics.blockBytes;
                used += heap.statistics.allocationBytes;
            }
            const auto free_block = ::Eden::Performance::graphics_memory_free.load(std::memory_order_relaxed);
            std::size_t guest_span = 0, guest_held = 0;
            ::Common::GuestMemoryUsage(&guest_span, &guest_held);
            std::fprintf(stderr, "EDEN_GRAPHICS_MEMORY refused outside the budget too (result %d): the pool's largest free "
                         "block is %llu MiB; the allocator holds %llu MiB in blocks, %llu MiB of it in use; the guest's "
                         "memory holds %llu MiB, the heap %llu MiB\\n",
                         int(created), free_block ? free_block() >> 20 : 0ull, blocks >> 20, used >> 20,
                         static_cast<unsigned long long>(guest_held >> 20),
                         static_cast<unsigned long long>(eden_heap_committed() >> 20));
            ::Eden::Performance::ReportHeap("a buffer was refused");
            if (ReleaseRefused("a buffer was refused", ci.size) != 0) {
                created = AGAIN;
                std::fprintf(stderr, "EDEN_GRAPHICS_MEMORY asked once more: result %d, DETAIL\\n", int(created), VALUES);
            }
            // The pool governor (src/direct_memory.c): a buffer has no fallback, so rather than end the
            // session it waits for memory while the reclaimers give back and the caches evict, up to
            // EDEN_GOV_BUFFER_WAIT_MS (default 3000; 0: end as before), and is asked for again each time
            // the pool has room for it.
            if (created != VK_SUCCESS && eden_governor_wait && eden_governor_wait_ms) {
                const unsigned wait_ms = eden_governor_wait_ms("EDEN_GOV_BUFFER_WAIT_MS", 3000);
                unsigned waited = 0;
                for (; created != VK_SUCCESS && waited < wait_ms; waited += 250) {
                    if (!eden_governor_wait(static_cast<std::size_t>(ci.size), 250, 3)) continue;
                    created = AGAIN;
                }
                std::fprintf(stderr, "EDEN_GOV buffer of %llu KiB %s after waiting %u ms\\n",
                             static_cast<unsigned long long>(ci.size >> 10),
                             created == VK_SUCCESS ? "placed" : "still refused", waited);
            }
            if (created != VK_SUCCESS) {
                ReportRefused();
            }
        }
    }
    vk::Check(created);
"""
def refused(call, detail, values):
    return (ALLOCATOR_REFUSED.replace('CALL', call).replace('AGAIN', call.replace('&alloc_ci', '&outside_budget'))
            .replace('DETAIL', detail).replace('VALUES', values))
IMAGE_VALUES = 'ci.extent.width, ci.extent.height, ci.extent.depth, ci.arrayLayers, ci.mipLevels, unsigned(ci.samples), int(ci.format)'
BUFFER_VALUES = 'static_cast<unsigned long long>(ci.size), unsigned(ci.usage)'
adapt('src/video_core/vulkan_common/vulkan_memory_allocator.h',
      'include/video_core/vulkan_common/vulkan_memory_allocator.h', [
    ('#pragma once', '#pragma once\n#include <array>\n#include <mutex>'),
    ('        vk::Image CreateImage(const VkImageCreateInfo &ci, bool dedicated = false) const;',
     '        vk::Image CreateImage(const VkImageCreateInfo &ci, bool dedicated = false) const;\n'
     '        vk::Image CreateTemporaryImage(const VkImageCreateInfo &ci) const;\n\n'
     '        /// The way to what the caches hold, for an allocation the pool refused: with `release`,\n'
     '        /// lets go of what the GPU is done with and gives the bytes; without, reports what they\n'
     '        /// hold. Set by the rasterizer, on any thread that is refused.\n'
     '        using RefusalHook = u64 (*)(void* context, bool release);\n'
     '        void SetRefusalHook(RefusalHook hook, void* context) noexcept;'),
    ('        static bool IsAutoUsage(VmaMemoryUsage u) noexcept {',
     '        u64 ReleaseRefused(const char* why, u64 wanted) const;\n'
     '        void ReportRefused() const;\n'
     '        bool NoFallbackLeft(const VkImageCreateInfo& ci) const;\n\n'
     '        static bool IsAutoUsage(VmaMemoryUsage u) noexcept {'),
    ('        u32 valid_memory_types{~0u};',
     '        u32 valid_memory_types{~0u};\n'
     '        mutable std::mutex temporary_pool_mutex;\n'
     '        mutable std::array<VmaPool, VK_MAX_MEMORY_TYPES> temporary_pools{};\n'
     '        std::atomic<RefusalHook> refusal_hook{nullptr};\n'
     '        void* refusal_context{};\n'
     '        /// Images the pool found nothing for, and since when (NoFallbackLeft).\n'
     '        mutable std::atomic<u32> images_refused{0};\n'
     '        mutable std::atomic<long long> images_refused_since{0};'),
])
adapt('src/video_core/vulkan_common/vulkan_memory_allocator.cpp', 'vulkan_memory_allocator.cpp', [
    ('            vmaCalculateStatistics(allocator, &statistics);',
     '            vmaCalculateStatistics(allocator, &statistics);\n'
     '            if (probe_rows == 160 || probe_rows == 256 || probe_rows == 512 || result != VK_SUCCESS)\n'
     '                EdenVmaSnapshot(allocator, probe_rows);'),
    ('vk::Image MemoryAllocator::CreateUnbackedImage(const VkImageCreateInfo &ci) const {',
     (port / 'transient_images.inc').read_text() + '\nvk::Image MemoryAllocator::CreateUnbackedImage(const VkImageCreateInfo &ci) const {'),
    ('MemoryAllocator::MemoryAllocator(const Device &device_)',
     r'''namespace {
std::mutex vma_probe_mutex;
VmaAllocator vma_probe_allocator{};
void EdenVmaSnapshot(VmaAllocator allocator, unsigned tag) {
    char* json = nullptr;
    vmaBuildStatsString(allocator, &json, VK_TRUE);
    if (!json) return;
    char path[192];
    std::snprintf(path, sizeof(path), "/data/eglantineps5/logs/mkw-vma-%d-%u.json", getpid(), tag);
    const auto bytes = std::strlen(json);
    bool written = false;
    if (bytes <= (8U << 20)) {
        if (FILE* file = std::fopen(path, "w")) {
            written = std::fwrite(json, 1, bytes, file) == bytes;
            std::fclose(file);
        }
    }
    std::fprintf(stderr, "MKW_VMA_MAP tag=%u bytes=%zu written=%u\n", tag, bytes, unsigned(written));
    vmaFreeStatsString(allocator, json);
}
}
extern "C" void eden_vma_probe(unsigned seconds) {
    std::lock_guard lock{vma_probe_mutex};
    if (vma_probe_allocator) EdenVmaSnapshot(vma_probe_allocator, 100000 + seconds);
}
// The budget's figures: atomics and no walk of the blocks, so any thread may ask, a refused one too.
// The allocator is externally synchronized: past 30 operations this also refreshes its budget,
// plain fields written without a lock, which the next refresh writes over.
extern "C" void eden_vma_usage(unsigned long long* blocks, unsigned long long* used) {
    std::lock_guard lock{vma_probe_mutex};
    if (!vma_probe_allocator) return;
    VmaBudget held[VK_MAX_MEMORY_HEAPS]{};
    vmaGetHeapBudgets(vma_probe_allocator, held);
    for (const VmaBudget& heap : held) {
        *blocks += heap.statistics.blockBytes;
        *used += heap.statistics.allocationBytes;
    }
}
MemoryAllocator::MemoryAllocator(const Device &device_)'''),
    ('    // Preserve the previous "RenderDoc small heap"',
     '    { std::lock_guard lock{vma_probe_mutex}; vma_probe_allocator = allocator; }\n'
     '    // Preserve the previous "RenderDoc small heap"'),
    ('MemoryAllocator::~MemoryAllocator() = default;',
     'MemoryAllocator::~MemoryAllocator() {\n'
     '    std::lock_guard lock{vma_probe_mutex};\n'
     '    if (vma_probe_allocator == allocator) vma_probe_allocator = nullptr;\n'
     '    for (auto pool : temporary_pools) if (pool) vmaDestroyPool(allocator, pool);\n' +
     ('    DestroySparsePools();\n' if 'DestroySparsePools' in (source / 'src/video_core/vulkan_common/vulkan_memory_allocator.h').read_text() else '') +
     '''}

void MemoryAllocator::SetRefusalHook(RefusalHook hook, void* context) noexcept {
    refusal_context = context;
    refusal_hook.store(hook, std::memory_order_release);
}

u64 MemoryAllocator::ReleaseRefused(const char* why, u64 wanted) const {
    const auto hook = refusal_hook.load(std::memory_order_acquire);
    const u64 staging = hook ? hook(refusal_context, true) : 0;
    ::Eden::Performance::graphics_refusals.fetch_add(1, std::memory_order_relaxed);
    const u64 heap = ::Eden::Performance::TrimHeap(why);
    // Nothing the caches hold gets a session past this moment: an image the collector deletes keeps
    // its memory until the next frame at the earliest. The guest's reserve is memory the pool can have
    // at once, down to its floor. A buffer only (wanted 0: an image, which has its fallback), up to
    // twice the request, at most 256 MiB.
    const u64 lend = (std::min)((std::max)(wanted * 2, u64{32} << 20), u64{256} << 20);
    const u64 lent = wanted != 0 && staging + heap < wanted && eden_guest_reserve_lend ? eden_guest_reserve_lend(lend) : 0;
    // Then cold guest memory, sent out now (EDEN_GUEST_SWAP_RECLAIM=1 with the swap on; after Aflakon's
    // patch 1905): a scene load asks for hundreds of MiB within seconds, before the swapper's next tick.
    // For images too: the pool's 2 MiB blocks side by side are what an image needs.
    const u64 reclaimed = staging + heap + lent < lend && eden_guest_memory_reclaim ? eden_guest_memory_reclaim(lend) : 0;
    std::fprintf(stderr, "EDEN_GRAPHICS_MEMORY %s: %llu MiB of finished staging, %llu MiB of the heap, %llu MiB of the "
                 "guest's reserve and %llu MiB of cold guest memory went back\\n",
                 why, static_cast<unsigned long long>(staging >> 20), static_cast<unsigned long long>(heap >> 20),
                 static_cast<unsigned long long>(lent >> 20), static_cast<unsigned long long>(reclaimed >> 20));
    return staging + heap + lent + reclaimed;
}

// What held the pool when it refused: the pool, the guest's memory, the allocator's blocks by memory
// type, then the caches. The periodic reports miss a burst that takes seconds. It takes no large
// block of its own, because the pool has none to give (so no vmaBuildStatsString here).
void MemoryAllocator::ReportRefused() const {
    if (!::Eden::Performance::RefusalReportDue()) {
        return;
    }
    ::Eden::Performance::ReportPool();
    VmaTotalStatistics held{};
    vmaCalculateStatistics(allocator, &held);
    for (u32 type = 0; type < properties.memoryTypeCount; ++type) {
        const VmaDetailedStatistics& kind = held.memoryType[type];
        if (kind.statistics.blockCount == 0) {
            continue;
        }
        std::fprintf(stderr, "EDEN_REFUSED_VMA type %u (flags 0x%x): %u blocks %llu MiB, %u allocations %llu MiB, the "
                     "largest %llu MiB; %u holes, the largest %llu MiB\\n",
                     type, unsigned(properties.memoryTypes[type].propertyFlags), kind.statistics.blockCount,
                     static_cast<unsigned long long>(kind.statistics.blockBytes >> 20), kind.statistics.allocationCount,
                     static_cast<unsigned long long>(kind.statistics.allocationBytes >> 20),
                     static_cast<unsigned long long>(kind.allocationSizeMax >> 20), kind.unusedRangeCount,
                     static_cast<unsigned long long>(kind.unusedRangeSizeMax >> 20));
    }
    if (const auto hook = refusal_hook.load(std::memory_order_acquire)) {
        hook(refusal_context, false);
    }
}

// Whether a refused image ends the session instead of going to the fallback with nothing bound.
// EDEN_NX2_UNBACKED_LIMIT=<images> in 2 s, 0: never.
bool MemoryAllocator::NoFallbackLeft(const VkImageCreateInfo& ci) const {
    static const u32 limit = [] {
        const char* value = std::getenv("EDEN_NX2_UNBACKED_LIMIT");
        return value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 10)) : 8U;
    }();
    const long long now = ::Eden::Performance::NowNs();
    if (now - images_refused_since.load(std::memory_order_relaxed) > 2'000'000'000) {
        images_refused_since.store(now, std::memory_order_relaxed);
        images_refused.store(0, std::memory_order_relaxed);
    }
    const u32 refused = images_refused.fetch_add(1, std::memory_order_relaxed) + 1;
    const bool ends = limit != 0 && refused >= limit;
    std::fprintf(stderr, "EDEN_GRAPHICS_MEMORY image %ux%ux%u, %u layers, %u levels, format %d found no memory: %u in 2 s, "
                 "limit %u%s\\n", ci.extent.width, ci.extent.height, ci.extent.depth, ci.arrayLayers, ci.mipLevels,
                 int(ci.format), refused, limit, ends ? ", the session ends" : "");
    return ends;
}'''),
    ('#include "video_core/vulkan_common/vulkan_memory_allocator.h"\n',
     '#include "video_core/vulkan_common/vulkan_memory_allocator.h"\n#include <atomic>\n#include <cstddef>\n#include <cstdio>\n#include <cstring>\n#include <cstdlib>\n#include "performance.h"\nnamespace Common { void GuestMemoryUsage(std::size_t* span, std::size_t* held) noexcept; }\nextern "C" std::size_t eden_heap_committed(void);\nextern "C" std::size_t eden_guest_reserve_lend(std::size_t) __attribute__((weak));\nextern "C" std::size_t eden_guest_memory_reclaim(std::size_t) __attribute__((weak));\nextern "C" int eden_governor_wait(std::size_t, unsigned, int) __attribute__((weak));\nextern "C" unsigned eden_governor_wait_ms(const char*, unsigned) __attribute__((weak));\n'),
    # An image the pool could not hold would be made with nothing bound: reads give zero, writes are
    # lost, and Eden's comment above says AMD's drivers do not survive sampling one. So it gets what a
    # buffer gets, and only then the fallback. The fallback stays: now and then an image too big for
    # what is left is no sign of the end (Ocarina, 2026-11-05: a 288 MiB depth image at 426 s, and the
    # session went on for 235 s). Many in a short time are: small images can still fit while each
    # frame's 13 MiB depth buffer does not (11 in the session's last second), and a session a retry
    # keeps alive would go on drawing into images with nothing behind them. The collector gets those
    # frames to free memory; then the session ends as it does for a buffer.
    ('    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && ci.imageType != VK_IMAGE_TYPE_1D) {\n',
     '''    bool no_fallback = false;
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && !huge) {
        if (ReleaseRefused("an image was refused", 0) != 0) {
            VmaAllocationCreateInfo once_more = alloc_ci;
            once_more.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
            result = vmaCreateImage(allocator, &ci, &once_more, &handle, &allocation, &alloc_info);
        }
        if (result != VK_SUCCESS) {
            no_fallback = NoFallbackLeft(ci);
            ReportRefused();
        }
    }
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && ci.imageType != VK_IMAGE_TYPE_1D && !no_fallback) {
'''),
    # What an image was that the allocator could not make at all, for stderr.log.
    ('    if (result != VK_SUCCESS) {\n        LOG_CRITICAL(Render_Vulkan, "Image {}x{}x{} ({} levels, {} layers, format {}) failed: {}",',
     """    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "EDEN_IMAGE_REFUSED %ux%ux%u, %u layers, %u levels, format %d, type %d, usage 0x%x, flags 0x%x, "
                     "tiling %d, samples %d, result %d, huge %d, texels %llu\\n",
                     ci.extent.width, ci.extent.height, ci.extent.depth, ci.arrayLayers, ci.mipLevels, int(ci.format),
                     int(ci.imageType), unsigned(ci.usage), unsigned(ci.flags), int(ci.tiling), int(ci.samples),
                     int(result), int(huge), static_cast<unsigned long long>(texels));
        LOG_CRITICAL(Render_Vulkan, "Image {}x{}x{} ({} levels, {} layers, format {}) failed: {}","""),
    # An image with nothing bound (the sparse virtual texture arrays no pool could hold): when the driver refuses
    # it with residency, it is asked for again with binding alone, and what it said is written down.
    ('    if (device.GetDispatchLoader().vkCreateImage(*device.GetLogical(), &sparse_ci, nullptr,\n'
     '                                                 &sparse) != VK_SUCCESS) {\n'
     '        return vk::Image{};\n'
     '    }\n',
     """    VkResult sparse_result = device.GetDispatchLoader().vkCreateImage(*device.GetLogical(), &sparse_ci, nullptr, &sparse);
    if (sparse_result != VK_SUCCESS) {
        VkImageCreateInfo binding_only = ci;
        binding_only.flags |= VK_IMAGE_CREATE_SPARSE_BINDING_BIT;
        const VkResult second = device.GetDispatchLoader().vkCreateImage(*device.GetLogical(), &binding_only, nullptr, &sparse);
        std::fprintf(stderr, "EDEN_SPARSE_IMAGE %ux%ux%u, %u layers, %u levels, format %d, usage 0x%x, flags 0x%x, tiling %d: "
                     "with residency %d, binding alone %d\\n",
                     ci.extent.width, ci.extent.height, ci.extent.depth, ci.arrayLayers, ci.mipLevels, int(ci.format),
                     unsigned(ci.usage), unsigned(ci.flags), int(ci.tiling), int(sparse_result), int(second));
        if (second != VK_SUCCESS) {
            // Once: which smaller sizes does the driver accept? Each probe is made and destroyed at once.
            static std::atomic<bool> probed{false};
            if (!probed.exchange(true)) {
                for (u32 layers = ci.arrayLayers / 2; layers >= 1; layers /= 2) {
                    VkImageCreateInfo probe = sparse_ci;
                    probe.arrayLayers = layers;
                    VkImage probe_image{};
                    const VkResult probed_result =
                        device.GetDispatchLoader().vkCreateImage(*device.GetLogical(), &probe, nullptr, &probe_image);
                    std::fprintf(stderr, "EDEN_SPARSE_PROBE %u layers: %d\\n", layers, int(probed_result));
                    if (probed_result == VK_SUCCESS) {
                        device.GetDispatchLoader().vkDestroyImage(*device.GetLogical(), probe_image, nullptr);
                    }
                    if (layers == 1) break;
                }
            }
            return vk::Image{};
        }
    }
"""),
    ('    vk::Check(vmaCreateBuffer(allocator, &ci, &alloc_ci, &handle, &allocation, &alloc_info));\n',
     refused('vmaCreateBuffer(allocator, &ci, &alloc_ci, &handle, &allocation, &alloc_info)',
             'buffer of %llu bytes, usage 0x%x', BUFFER_VALUES)),
    ('    vk::Check(vmaCreateBufferWithAlignment(allocator, &ci, &alloc_ci, min_alignment, &handle,\n'
     '                                           &allocation, &alloc_info));\n',
     refused('vmaCreateBufferWithAlignment(allocator, &ci, &alloc_ci, min_alignment, &handle,\n'
             '                                           &allocation, &alloc_info)',
             'buffer of %llu bytes, usage 0x%x', BUFFER_VALUES)),
])
# Staging buffers (headless/staging_buffers.h): above 16 MiB a buffer the CPU writes or reads is its
# transfer rounded up to 16 MiB instead of the next power of two, a kept buffer is handed out again
# only when it holds the transfer, and a refused buffer is asked for once more after the kept ones
# the GPU is done with are let go. A Switch 2 title was refused 512 MiB for a 341 MiB transfer with
# 480 MiB free. No Finish() when releasing: a buffer handed out earlier may not have its copy recorded.
adapt('src/video_core/renderer_vulkan/vk_staging_buffer_pool.h',
      'include/video_core/renderer_vulkan/vk_staging_buffer_pool.h', [
    ('    void TickFrame();\n',
     '    void TickFrame();\n    std::uint64_t ReleaseCompleted();\n    void ReportHeld();\n'),
])
adapt('src/video_core/renderer_vulkan/vk_staging_buffer_pool.cpp', 'vulkan_staging_buffer_pool.cpp', [
    ('StagingBufferRef StagingBufferPool::CreateStagingBuffer(size_t size, MemoryUsage usage, bool deferred) {',
     '''StagingBufferRef StagingBufferPool::CreateStagingBuffer(size_t size, MemoryUsage usage, bool deferred) {
    if (static unsigned rows{}; size >= (64ULL << 20) && rows++ < 96) {
        for (unsigned kind = 0; kind < 3; ++kind) {
            auto& cache = kind == 0 ? device_local_cache : kind == 1 ? upload_cache : download_cache;
            size_t count = 0, pinned = 0, busy = 0, free = 0;
            unsigned long long bytes = 0;
            for (const auto& level : cache) for (const auto& entry : level.entries) {
                ++count;
                bytes += ::Eden::StagingBuffers::Holds(entry.mapped_span.size(), entry.log2_level);
                if (entry.deferred) ++pinned;
                else if (scheduler.IsFree(entry.tick)) ++free;
                else ++busy;
            }
            std::fprintf(stderr, "MKW_STAGING_MISS asked=%zu usage=%u deferred=%d kind=%u count=%zu bytes=%llu pinned=%zu busy=%zu free=%zu tick=%llu completed=%llu\\n",
                         size, unsigned(usage), int(deferred), kind, count, bytes, pinned, busy, free,
                         static_cast<unsigned long long>(scheduler.CurrentTick()),
                         static_cast<unsigned long long>(scheduler.GetMasterSemaphore().KnownGpuTick()));
        }
    }
'''),
    ('#include "video_core/vulkan_common/vulkan_wrapper.h"\n',
     '#include "video_core/vulkan_common/vulkan_wrapper.h"\n#include <atomic>\n#include <chrono>\n#include <cstdio>\n'
     '#include "performance.h"\n#include "staging_buffers.h"\n'),
    ('void StagingBufferPool::TickFrame() {',
     '''void StagingBufferPool::ReportHeld() {
    const char* name = "device";
    for (StagingBuffersCache* const cache : {&device_local_cache, &upload_cache, &download_cache}) {
        size_t count = 0, pinned = 0, busy = 0;
        unsigned long long bytes = 0, idle = 0;
        for (const StagingBuffers& level : *cache) {
            for (const StagingBuffer& entry : level.entries) {
                const auto held = ::Eden::StagingBuffers::Holds(entry.mapped_span.size(), entry.log2_level);
                ++count;
                bytes += held;
                if (entry.deferred) {
                    ++pinned;
                } else if (scheduler.IsFree(entry.tick)) {
                    idle += held;
                } else {
                    ++busy;
                }
            }
        }
        std::fprintf(stderr, "EDEN_REFUSED_STAGING %s: %zu buffers hold %llu MiB, %llu MiB finished; %zu pinned, %zu with "
                     "the GPU\\n", name, count, bytes >> 20, idle >> 20, pinned, busy);
        name = cache == &device_local_cache ? "upload" : "download";
    }
    std::fprintf(stderr, "EDEN_REFUSED_STAGING stream: %llu MiB\\n",
                 static_cast<unsigned long long>(stream_buffer_size >> 20));
}

'''
     'std::uint64_t StagingBufferPool::ReleaseCompleted() {\n'
     '    const auto idle = [this](const StagingBuffer& entry) {\n'
     '        return !entry.deferred && scheduler.IsFree(entry.tick);\n'
     '    };\n'
     '    ::Eden::StagingBuffers::Released released;\n'
     '    for (StagingBuffersCache* const cache : {&device_local_cache, &upload_cache, &download_cache}) {\n'
     '        for (StagingBuffers& level : *cache) {\n'
     '            released += ::Eden::StagingBuffers::ReleaseFrom(level, idle);\n'
     '        }\n'
     '    }\n'
     '    return released.bytes;\n'
     '}\n\nvoid StagingBufferPool::TickFrame() {\n'
     '    // While the pool the CPU shares with the GPU is short, kept buffers the GPU is done with go\n'
     '    // back to it, at most every half second: an upload buffer kept for the next transfer is\n'
     '    // memory the game\'s writes, the heap or the JIT can need first (Mario Kart World kept 218 MiB).\n'
     '    if (::Eden::Performance::graphics_memory_short.load(std::memory_order_relaxed)) {\n'
     '        static std::chrono::steady_clock::time_point last_release{};\n'
     '        const auto now = std::chrono::steady_clock::now();\n'
     '        if (now - last_release >= std::chrono::milliseconds{500}) {\n'
     '            last_release = now;\n'
     '            ReleaseCompleted();\n'
     '        }\n'
     '    }'),
    ('    const auto is_free = [this](const StagingBuffer& entry) {\n'
     '        return !entry.deferred && scheduler.IsFree(entry.tick);\n'
     '    };',
     '    const auto is_free = [this, size](const StagingBuffer& entry) {\n'
     '        return !entry.deferred && scheduler.IsFree(entry.tick) &&\n'
     '               ::Eden::StagingBuffers::Holds(entry.mapped_span.size(), entry.log2_level) >= size;\n'
     '    };'),
    ('    auto const log2_size = Common::Log2Ceil<u32>(u32(size));\n',
     '    auto const log2_size = Common::Log2Ceil<u32>(u32(size));\n'
     '    const bool exact = ::Eden::StagingBuffers::Exact(size, usage != MemoryUsage::DeviceLocal);\n'
     '    const VkDeviceSize made = ::Eden::StagingBuffers::Size(size, log2_size, exact);\n'
     '    if (exact) {\n'
     '        // Kept buffers of this power of two too small for the transfer go before it is made.\n'
     '        ::Eden::StagingBuffers::ReleaseFrom(GetCache(usage)[log2_size], [this, size](const StagingBuffer& entry) {\n'
     '            return !entry.deferred && scheduler.IsFree(entry.tick) &&\n'
     '                   ::Eden::StagingBuffers::Holds(entry.mapped_span.size(), entry.log2_level) < size;\n'
     '        });\n'
     '    }\n'),
    ('        .size = 1ULL << log2_size,\n', '        .size = made,\n'),
    ('    vk::Buffer buffer = memory_allocator.CreateBuffer(buffer_ci, usage);\n',
     '    vk::Buffer buffer;\n'
     '    try {\n'
     '        buffer = memory_allocator.CreateBuffer(buffer_ci, usage);\n'
     '    } catch (const vk::Exception& refused) {\n'
     '        if (refused.GetResult() != VK_ERROR_OUT_OF_DEVICE_MEMORY) {\n'
     '            throw;\n'
     '        }\n'
     '        // Refused: the kept buffers the GPU is done with go, and the buffer is asked for once\n'
     '        // more. Refused again, the error goes on as before.\n'
     '        const auto idle = [this](const StagingBuffer& entry) {\n'
     '            return !entry.deferred && scheduler.IsFree(entry.tick);\n'
     '        };\n'
     '        ::Eden::StagingBuffers::Released released;\n'
     '        for (StagingBuffersCache* const cache : {&device_local_cache, &upload_cache, &download_cache}) {\n'
     '            for (StagingBuffers& level : *cache) {\n'
     '                released += ::Eden::StagingBuffers::ReleaseFrom(level, idle);\n'
     '            }\n'
     '        }\n'
     '        std::fprintf(stderr, "EDEN_STAGING_OOM asked=%zu made=%llu usage=%d deferred=%d released=%zu buffers (%llu MiB)\\n",\n'
     '                     size, static_cast<unsigned long long>(made), int(usage), int(deferred), released.buffers,\n'
     '                     static_cast<unsigned long long>(released.bytes >> 20));\n'
     '        // Eight MiB and a page took 16 MiB, and the pool had 13 in one piece. Only what the CPU maps:\n'
     '        // Holds() knows the size of those; a kept buffer of the others is taken to be its power of two.\n'
     '        // In steps of 2 MiB, what a mapping takes anyway: less would only leave dust in the pool.\n'
     '        if (usage != MemoryUsage::DeviceLocal) {\n'
     '            buffer_ci.size = (std::min<VkDeviceSize>)(made, Common::AlignUp(VkDeviceSize(size), VkDeviceSize{2} << 20));\n'
     '        }\n'
     '        buffer = memory_allocator.CreateBuffer(buffer_ci, usage);\n'
     '    }\n'
     '    // A large buffer smaller than the power of two: the 1st, 2nd, 4th, 8th ... in the log.\n'
     '    if (exact && made >= (VkDeviceSize{64} << 20) && made < (VkDeviceSize{1} << log2_size)) {\n'
     '        static std::atomic<unsigned long long> large{0};\n'
     '        const auto times = large.fetch_add(1, std::memory_order_relaxed) + 1;\n'
     '        if ((times & (times - 1)) == 0) {\n'
     '            std::fprintf(stderr, "EDEN_STAGING_EXACT %llu MiB made for a transfer of %llu MiB, not the %llu MiB of the "\n'
     '                         "next power of two (%llu so far)\\n",\n'
     '                         static_cast<unsigned long long>(made >> 20), static_cast<unsigned long long>(size >> 20),\n'
     '                         static_cast<unsigned long long>((VkDeviceSize{1} << log2_size) >> 20), times);\n'
     '        }\n'
     '    }\n'),
])
# Development A/B switches for device features (headless/dev_vulkan.h, dev-settings).
adapt('src/video_core/vulkan_common/vulkan_device.cpp', 'vulkan_device.cpp', [
    ('#include <algorithm>', '#include <algorithm>\n#include <cstdio>\n#include "dev_vulkan.h"\n#include "performance.h"'),
    # "Memory in use" for the texture and buffer caches: on the console, the budget less what is
    # left of the pool the CPU shares with the GPU (performance.h, graphics_memory_free).
    ('u64 Device::GetDeviceMemoryUsage() const {\n',
     'namespace {\n'
     '// The largest block of the shared pool that was free when the device was made: about the most\n'
     '// graphics can be given in this session.\n'
     'u64 graphics_capacity = 0;\n'
     '}\n\n'
     'u64 Device::GetDeviceMemoryUsage() const {\n'
     '    if (const auto free_memory = ::Eden::Performance::graphics_memory_free.load(std::memory_order_relaxed);\n'
     '        free_memory && ::Eden::Performance::graphics_usage_from_pool.load(std::memory_order_relaxed)) {\n'
     '        // The caches evict what is unused 1.6 GiB under the budget and evict hard 0.8 GiB under\n'
     '        // it. Where the pool could never give graphics 6.4 GiB, those marks become a quarter\n'
     '        // and an eighth of what it could give.\n'
     '        const double scale = graphics_capacity != 0 && graphics_capacity < 6400_MiB\n'
     '                                 ? static_cast<double>(6400_MiB) / static_cast<double>(graphics_capacity)\n'
     '                                 : 1.0;\n'
     '        const u64 free_bytes = static_cast<u64>(static_cast<double>(free_memory()) * scale);\n'
     '        return device_access_memory > free_bytes ? device_access_memory - free_bytes : 0;\n'
     '    }\n'),
    # Usage above is expressed against device_access_memory, not the driver's
    # changing heap budget. Keep both sides of the cache's pressure comparison
    # in that same domain; the shared-pool free callback already reflects CPU
    # allocations and fragmentation.
    ('u64 Device::GetDeviceMemoryBudget() const {\n',
     'u64 Device::GetDeviceMemoryBudget() const {\n'
     '    if (::Eden::Performance::graphics_memory_free.load(std::memory_order_relaxed) &&\n'
     '        ::Eden::Performance::graphics_usage_from_pool.load(std::memory_order_relaxed)) {\n'
     '        return device_access_memory;\n'
     '    }\n'),
    ('    extensions.descriptor_buffer = features.descriptor_buffer.descriptorBuffer;',
     '    extensions.descriptor_buffer = features.descriptor_buffer.descriptorBuffer &&\n'
     '        !::Eden::DevVulkan::disable_descriptor_buffer;'),
    ('    features.robustness2.robustBufferAccess2 = VK_FALSE;\n'
     '    features.robustness2.robustImageAccess2 = VK_FALSE;\n'
     '    extensions.robustness_2 = features.robustness2.nullDescriptor;',
     '    if (!::Eden::DevVulkan::robustness2) {\n'
     '        // EDEN_NX2_ROBUST_BUFFERS=1 (mja00/Lichii 6571c04166): out-of-bounds storage buffer\n'
     '        // reads return garbage on AMD without it; Ocarina fog compute shader\n'
     '        // 75d4369f3b1f5868 filled distant froxels with large values and turned the sky white.\n'
     '        const char* const robust_buffers = std::getenv("EDEN_NX2_ROBUST_BUFFERS");\n'
     '        if (robust_buffers == nullptr || robust_buffers[0] == \'0\') {\n'
     '            features.robustness2.robustBufferAccess2 = VK_FALSE;\n'
     '        }\n'
     '        features.robustness2.robustImageAccess2 = VK_FALSE;\n'
     '    }\n'
     '    if (::Eden::DevVulkan::disable_null_descriptor) {\n'
     '        features.robustness2.nullDescriptor = VK_FALSE;\n'
     '    }\n'
     '    extensions.robustness_2 = features.robustness2.nullDescriptor;\n'
     '    LOG_INFO(Render_Vulkan, "robustness2 extension {} robustBufferAccess2 {} nullDescriptor {}",\n'
     '             extensions.robustness_2, features.robustness2.robustBufferAccess2 != VK_FALSE,\n'
     '             features.robustness2.nullDescriptor != VK_FALSE);'),
    ('            (queue_family_properties[*graphics].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) != 0;',
     '            (queue_family_properties[*graphics].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) != 0 &&\n'
     '            !::Eden::DevVulkan::disable_sparse;'),
    ('            features.custom_border_color.customBorderColorWithoutFormat;\n    }\n'
     '    RemoveExtensionFeatureIfUnsuitable(extensions.custom_border_color',
     '            features.custom_border_color.customBorderColorWithoutFormat &&\n'
     '            !::Eden::DevVulkan::disable_custom_border;\n    }\n'
     '    RemoveExtensionFeatureIfUnsuitable(extensions.custom_border_color'),
    # dev-settings conditional_rendering=off: guest render-enable conditions are resolved
    # without VK_EXT_conditional_rendering (the path Eden takes on drivers without it).
    ('    // VK_EXT_border_color_swizzle\n',
     '    if (::Eden::DevVulkan::disable_conditional_rendering) {\n'
     '        extensions.conditional_rendering = false;\n    }\n'
     '    // VK_EXT_border_color_swizzle\n'),
    # The heaps and the budget the texture/buffer caches size themselves from.
    ('            device_access_memory = std::min<u64>(device_access_memory, normal_memory + scaler_memory);\n'
     '        }\n    }\n}\n',
     '            device_access_memory = std::min<u64>(device_access_memory, normal_memory + scaler_memory);\n'
     '        }\n    }\n'
     '    if (const auto free_memory = ::Eden::Performance::graphics_memory_free.load(std::memory_order_relaxed)) {\n'
     '        graphics_capacity = free_memory();\n'
     '    }\n'
     '    std::printf("EDEN_VULKAN_MEMORY integrated=%d heaps=%zu local=%llu initial_usage=%llu device_access=%llu "\n'
     '                "capacity=%llu\\n",\n'
     '                int(is_integrated), valid_heap_memory.size(), static_cast<unsigned long long>(local_memory),\n'
     '                static_cast<unsigned long long>(device_initial_usage),\n'
     '                static_cast<unsigned long long>(device_access_memory),\n'
     '                static_cast<unsigned long long>(graphics_capacity));\n}\n'),
])
# Development GPU fault probe (dev_vulkan.h, submit_sync=on): wait for every
# submission, so the CPU never records against work the GPU has not finished.
GPU_TIME_HELPERS = (port / 'gpu_time_probe.inc').read_text()
# EDEN_NX2_END_FAIL_SOFT=1: a command buffer the driver cannot end is dropped instead of ending
# the session. Run 102 (Mario Kart, T27): vkEndCommandBuffer answered VK_ERROR_OUT_OF_DEVICE_MEMORY
# at the course load, the exception left the worker thread, and the app stopped itself; run 101
# lived through the same shortage and raced. Two empty buffers carry the semaphores, so every wait
# on the tick completes. The dropped buffer's draws, uploads and downloads are lost: a wrong frame,
# and a readback that brings stale bytes to the guest, are possible. Off unless asked for.
END_FAIL_SOFT_HELPERS = """
void EndCommandBuffersOrDrop(vk::CommandBuffer& upload_cmdbuf, vk::CommandBuffer& cmdbuf,
                             CommandPool& command_pool, const Device& device, u64 signal_value) {
    static const bool end_fail_soft = [] {
        const char* const value = std::getenv("EDEN_NX2_END_FAIL_SOFT");
        return value != nullptr && value[0] == '1';
    }();
    try {
        upload_cmdbuf.End();
        cmdbuf.End();
        return;
    } catch (const vk::Exception& exception) {
        if (!end_fail_soft) {
            throw;
        }
        static u32 dropped{};
        LOG_ERROR(Render_Vulkan, "Command buffer could not be ended ({}): its work is dropped, tick {} is signalled empty ({} so far)",
                  exception.what(), signal_value, ++dropped);
    }
    static constexpr VkCommandBufferBeginInfo EMPTY_BEGIN{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    // The pool resets a buffer when it begins again, so the two that failed need nothing.
    upload_cmdbuf = vk::CommandBuffer(command_pool.Commit(), device.GetDispatchLoader());
    upload_cmdbuf.Begin(EMPTY_BEGIN);
    upload_cmdbuf.End();
    cmdbuf = vk::CommandBuffer(command_pool.Commit(), device.GetDispatchLoader());
    cmdbuf.Begin(EMPTY_BEGIN);
    cmdbuf.End();
}
"""
# Keep command availability separate from queue-drain completion. Only the worker
# waits on event_cv; WaitWorker waits on drain_cv under the same queue mutex.
adapt('src/video_core/renderer_vulkan/vk_scheduler.h',
      'include/video_core/renderer_vulkan/vk_scheduler.h', [
    ('    std::condition_variable_any event_cv;',
     '    std::condition_variable_any event_cv;\n    std::condition_variable drain_cv;\n    unsigned drain_waiters{};'),
])
adapt('src/video_core/renderer_vulkan/vk_scheduler.cpp', 'vulkan_scheduler.cpp', [
    ('Scheduler::~Scheduler() = default;',
     'Scheduler::~Scheduler() {\n'
     '    worker_thread.request_stop();\n'
     '    if (worker_thread.joinable()) worker_thread.join();\n'
     '    GpuTimeReset(device);\n}'),
    ('        event_cv.wait(ql, [this] { return work_queue.empty(); });',
     '        ++drain_waiters;\n        drain_cv.wait(ql, [this] { return work_queue.empty(); });\n        --drain_waiters;'),
    ('            event_cv.notify_all();', '            if (eden_vulkan_handoff_enabled()) event_cv.notify_one();\n            else event_cv.notify_all();'),
    ('        work_queue.pop();\n        event_cv.notify_all();',
     '        work_queue.pop();\n        if (!eden_vulkan_handoff_enabled() || (work_queue.empty() && drain_waiters)) drain_cv.notify_all();'),

    ('#include <memory>', '#include <memory>\n#include <array>\n#include <chrono>\n#include <cstdio>\n#include "dev_vulkan.h"\n#include "submission_origin.h"\nextern "C" bool eden_vulkan_handoff_enabled();'),
    ('namespace Vulkan {\n', 'namespace Vulkan {\n' + GPU_TIME_HELPERS + END_FAIL_SOFT_HELPERS),
    ('        .pInheritanceInfo = nullptr,\n    });\n}\n\nu64 Scheduler::SubmitExecution(',
     '        .pInheritanceInfo = nullptr,\n    });\n    GpuTimeBegin(device, current_upload_cmdbuf);\n}\n\nu64 Scheduler::SubmitExecution('),
    ('u64 Scheduler::Flush(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore) {',
     'u64 Scheduler::Flush(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore) {\n'
     '    GpuSubmitOriginScope origin{__builtin_return_address(0), false, true};'),
    ('void Scheduler::Finish(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore) {',
     'void Scheduler::Finish(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore) {\n'
     '    GpuSubmitOriginScope origin{__builtin_return_address(0), true, true};'),
    ('    RecordWithUploadBuffer([signal_semaphore, wait_semaphore, signal_value,',
     '    RecordWithUploadBuffer([signal_semaphore, wait_semaphore, signal_value,\n'
     '                            submit_origin = gpu_submit_origin,'),
    # EDEN_NX2_END_FAIL_SOFT (END_FAIL_SOFT_HELPERS): both ends go through the helper.
    ('        upload_cmdbuf.End();\n        cmdbuf.End();\n\n        if (on_submit)',
     '        GpuTimeEnd(cmdbuf);\n'
     '        EndCommandBuffersOrDrop(upload_cmdbuf, cmdbuf, *command_pool, device, signal_value);\n\n'
     '        if (on_submit)'),
    ('            vk::Check(result);\n            break;\n        }\n    });\n    chunk->MarkSubmit();',
     '            vk::Check(result);\n            break;\n        }\n'
     '        GpuTimeCountOrigin(submit_origin);\n        GpuTimeCollect(device);\n    });\n    chunk->MarkSubmit();'),
    ('    const u64 signal_value = SubmitExecution(signal_semaphore, wait_semaphore);\n'
     '    AllocateNewContext();\n    last_flush',
     '    const u64 signal_value = SubmitExecution(signal_semaphore, wait_semaphore);\n'
     '    if (::Eden::DevVulkan::sync_submissions) {\n        Wait(signal_value);\n    }\n'
     '    AllocateNewContext();\n    last_flush'),
])
adapt('src/video_core/renderer_vulkan/pipeline_helper.h',
      'include/video_core/renderer_vulkan/pipeline_helper.h', [
    ('            VkDescriptorGetInfoEXT get_info{',
     '            const VkDescriptorAddressInfoEXT* const buffer_info{\n'
     '                entry.address.address != 0 ? &address_info : nullptr};\n'
     '            VkDescriptorGetInfoEXT get_info{'),
    ('get_info.data.pUniformBuffer = &address_info;', 'get_info.data.pUniformBuffer = buffer_info;'),
    ('get_info.data.pStorageBuffer = &address_info;', 'get_info.data.pStorageBuffer = buffer_info;'),
    ('get_info.data.pUniformTexelBuffer = &address_info;', 'get_info.data.pUniformTexelBuffer = buffer_info;'),
    ('get_info.data.pStorageTexelBuffer = &address_info;', 'get_info.data.pStorageTexelBuffer = buffer_info;'),
])
adapt('src/video_core/renderer_vulkan/vk_blit_screen.h',
      'include/video_core/renderer_vulkan/vk_blit_screen.h', [
    ('VkFormat current_swapchain_view_format);',
     'VkFormat current_swapchain_view_format, bool draw_hud = false);'),
])
adapt('src/video_core/renderer_vulkan/renderer_vulkan.h',
      'include/video_core/renderer_vulkan/renderer_vulkan.h', [
    ('    void Composite(', '    void PresentLoading();\n    void Composite('),
])
adapt('src/video_core/renderer_vulkan/renderer_vulkan.cpp', 'vulkan_renderer.cpp', [
    ('#include "video_core/renderer_vulkan/renderer_vulkan.h"',
     '#include "video_core/renderer_vulkan/renderer_vulkan.h"\n#include "display_refresh.h"\n'
     '#include "diagnostics.h"\n#include "hud.h"'),
    # The HUD counted the guest rate only: OnFrameDisplayed() runs once per Composite(), while the
    # frames frame generation adds are presented through PresentManager::Present() (hud.h). Count
    # every frame handed to the present manager.
    ('        present_manager.Present(generated);\n    }',
     '        present_manager.Present(generated);\n'
     '        Eden::CountPresentedFrame();\n'
     '    }'),
    # A frame the display has no refresh for is not presented: a game patched for more frames
    # than the output shows keeps its pace instead of waiting for a refresh per frame
    # (headless/display_refresh.h). The frame still ends as any other.
    ('    RenderScreenshot(framebuffers);\n    Frame* frame = present_manager.GetRenderFrame();',
     '    RenderScreenshot(framebuffers);\n'
     '    if (::Eden::Display::SkipFrame()) {\n'
     '        gpu.RendererFrameEndNotify();\n'
     '        rasterizer.TickFrame();\n'
     '        return;\n'
     '    }\n'
     '    Frame* frame = present_manager.GetRenderFrame();'),
    ('    Report();', '    PresentLoading();\n    Report();'),
    ('swapchain.GetImageViewFormat());', 'swapchain.GetImageViewFormat(), true);'),
    ('layout, 1, format);', 'layout, 1, format, true);'),
    ('void RendererVulkan::Report() const {',
     (port / 'vulkan_loading.inc').read_text() + '\nvoid RendererVulkan::Report() const {'),
    # Present() only records the hand-off into a fresh chunk. Dispatch it now so
    # the present thread gets this frame immediately instead of after the
    # producer's next 8-draw dispatch (upstream does this only with LSFG).
    ('    present_manager.Present(frame);\n#ifdef HAS_LSFG\n    scheduler.DispatchWork();\n#endif',
     '    present_manager.Present(frame);\n    Eden::CountPresentedFrame();\n'
     '    scheduler.DispatchWork();'),
    ('} // namespace Vulkan', '''} // namespace Vulkan
namespace Eden {
void PresentVulkanLoading(VideoCore::RendererBase& renderer) {
    static_cast<Vulkan::RendererVulkan&>(renderer).PresentLoading();
}
}'''),
])
adapt('src/video_core/renderer_vulkan/present/window_adapt_pass.h',
      'include/video_core/renderer_vulkan/present/window_adapt_pass.h', [
    ('const Layout::FramebufferLayout& layout, Frame* dst);',
     'const Layout::FramebufferLayout& layout, Frame* dst, bool draw_hud = false);'),
    ('    vk::Pipeline coverage_pipeline;',
     '    vk::Pipeline coverage_pipeline;\n    vk::PipelineLayout hud_layout;\n    vk::Pipeline hud_pipeline;'),
])
adapt('src/video_core/renderer_vulkan/present/window_adapt_pass.cpp', 'vulkan_window_adapt_pass.cpp', [
    ('#include "core/frontend/framebuffer_layout.h"',
     '#include "core/frontend/framebuffer_layout.h"\n#include "hud.h"\n#include "vulkan_hud_shaders.h"'),
    ('const Layout::FramebufferLayout& layout, Frame* dst) {',
     'const Layout::FramebufferLayout& layout, Frame* dst, bool draw_hud) {'),
    ('    scheduler.Record([=](vk::CommandBuffer cmdbuf) {',
     (port / 'vulkan_hud_prepare.inc').read_text() + '\n    scheduler.Record([=](vk::CommandBuffer cmdbuf) {'),
    ('        cmdbuf.EndRenderPass();',
     (port / 'vulkan_hud_draw.inc').read_text() + '\n        cmdbuf.EndRenderPass();'),
])
adapt('src/video_core/renderer_vulkan/vk_blit_screen.cpp', 'vulkan_blit_screen.cpp', [
    ('VkFormat current_swapchain_view_format) {',
     'VkFormat current_swapchain_view_format, bool draw_hud) {'),
    ('image_index, layers, framebuffers, layout, frame);',
     'image_index, layers, framebuffers, layout, frame, draw_hud);'),
])
shader_header = '#pragma once\n#include <cstdint>\n'
for stage in ('vert', 'frag'):
    binary = output / f'vulkan_hud.{stage}.spv'
    subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.0',
                    '-o', str(binary), str(port / f'vulkan_hud.{stage}')], check=True)
    subprocess.run(['spirv-val', '--target-env', 'vulkan1.0', str(binary)], check=True)
    data = binary.read_bytes()
    words = struct.unpack('<' + 'I' * (len(data) // 4), data)
    shader_header += f'inline constexpr uint32_t EDEN_HUD_{stage.upper()}_SPV[] = {{\n'
    shader_header += ','.join(hex(word) for word in words) + '\n};\n'
path = output / 'vulkan_hud_shaders.h'
if not path.exists() or path.read_text() != shader_header:
    path.write_text(shader_header)
adapt('src/video_core/vulkan_common/vulkan.h', 'include/video_core/vulkan_common/vulkan.h',
      [('#else\n#define VK_USE_PLATFORM_XLIB_KHR', '#elif defined(PS5_NATIVE)\n// Native display-plane presentation, no desktop window-system headers.\n#else\n#define VK_USE_PLATFORM_XLIB_KHR')])
adapt('src/video_core/vulkan_common/vulkan_instance.cpp', 'vulkan_instance.cpp', [
    ('namespace Vulkan {', 'extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance, const char*);\n\nnamespace Vulkan {'),
    ('    case Core::Frontend::WindowSystemType::Headless:', '    case Core::Frontend::WindowSystemType::PS5:\n        extensions.push_back(VK_KHR_DISPLAY_EXTENSION_NAME);\n        break;\n    case Core::Frontend::WindowSystemType::Headless:'),
    ('#else\n    case Core::Frontend::WindowSystemType::X11:', '#elif !defined(PS5_NATIVE)\n    case Core::Frontend::WindowSystemType::X11:'),
    ('''    if (!library.IsOpen()) {
        LOG_ERROR(Render_Vulkan, "Vulkan library not available");
        throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
    }
    if (!library.GetSymbol("vkGetInstanceProcAddr", &dld.vkGetInstanceProcAddr)) {
        LOG_ERROR(Render_Vulkan, "vkGetInstanceProcAddr not present in Vulkan");
        throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
    }''', '    (void)library;\n    dld.vkGetInstanceProcAddr = &::vkGetInstanceProcAddr;'),
])
adapt('src/video_core/vulkan_common/vulkan_library.cpp', 'vulkan_library.cpp', [
    ('#if defined(__ANDROID__) && defined(ARCHITECTURE_arm64)', '#if defined(PS5_NATIVE)\n    return std::make_shared<Common::DynamicLibrary>();\n#elif defined(__ANDROID__) && defined(ARCHITECTURE_arm64)'),
])
adapt('src/video_core/vulkan_common/vulkan_wrapper.h', 'include/video_core/vulkan_common/vulkan_wrapper.h', [
    ('#include <array>', '#include <array>\n#include <cstdio>\n#include "performance.h"'),
    ('return dld->vkQueueSubmit(queue,', 'auto timer = Eden::Performance::VulkanTimer(2);\n        return dld->vkQueueSubmit(queue,'),
    ('return dld->vkQueueSubmit2(queue,', 'auto timer = Eden::Performance::VulkanTimer(2);\n        return dld->vkQueueSubmit2(queue,'),
    ('return dld->vkWaitForFences(owner,', 'auto timer = Eden::Performance::VulkanTimer(3);\n        return dld->vkWaitForFences(owner,'),
    ('const VkResult result = dld->vkWaitSemaphores(owner,', 'auto timer = Eden::Performance::VulkanTimer(4);\n        const VkResult result = dld->vkWaitSemaphores(owner,'),
    ('return dld->vkDeviceWaitIdle(handle);', 'auto timer = Eden::Performance::VulkanTimer(5);\n        return dld->vkDeviceWaitIdle(handle);'),
    ('''inline void Check(VkResult result) {
    if (result != VK_SUCCESS) {
        throw Exception(result);''', '''inline void Check(VkResult result, const char* file = __builtin_FILE(),
                  int line = __builtin_LINE(), const char* operation = __builtin_FUNCTION()) {
    if (result != VK_SUCCESS) {
        std::fprintf(stderr, "[EglantinePS5] Vulkan failure: %s:%d (%s) VkResult=%d\\n",
                     file, line, operation, static_cast<int>(result));
        throw Exception(result);'''),
])
adapt('src/video_core/vulkan_common/vulkan_wrapper.cpp', 'vulkan_wrapper.cpp', [
    ('#include "video_core/vulkan_common/vulkan_wrapper.h"', '#include "video_core/vulkan_common/vulkan_wrapper.h"\n#include "performance.h"'),
    ('auto const result = dld->vkCreateGraphicsPipelines(', 'auto timer = Eden::Performance::VulkanTimer(0);\n    auto const result = dld->vkCreateGraphicsPipelines('),
    ('Check(dld->vkCreateComputePipelines(', 'auto timer = Eden::Performance::VulkanTimer(1);\n    Check(dld->vkCreateComputePipelines('),
])
download_source = (source / 'src/video_core/texture_cache/texture_cache.h').read_text()
download_begin = download_source.index('void TextureCache<P>::DownloadMemory(DAddr cpu_addr, size_t size)')
download_loop = download_source.index('    for (const ImageId image_id : images) {', download_begin)
download_prefix = download_source[download_begin:download_loop]
# Reuse DeleteImage's exact accounting to stop lookahead when its existing
# pressure policy changes. Fail on source drift rather than inventing a budget.
delete_begin = download_source.index('    if (image.HasScaled()) {', download_source.index('void TextureCache<P>::DeleteImage('))
delete_end = download_source.index('    const GPUVAddr gpu_addr', delete_begin)
reclaimed_bytes = (download_source[delete_begin:delete_end]
                   .replace('total_used_memory -=', 'bytes +=')
                   .replace('slot_images[image_id]', 'image'))
assert 'image_id' not in reclaimed_bytes, 'DeleteImage accounting needs adapting to an image reference'
gc_begin = download_source.index('void TextureCache<P>::RunGarbageCollector() {')
gc_end = download_source.index('\ntemplate <class P>', gc_begin)
gc_original = download_source[gc_begin:gc_end]
gc = gc_original
POOL_PRESSURE = r'''    const auto PoolPressure = [this]() -> int {
        if constexpr (!std::is_same_v<Runtime, Vulkan::TextureCacheRuntime>) {
            return 0;
        } else {
            if (!::Eden::Performance::texture_usage_from_images.load(std::memory_order_relaxed)) {
                return 0;
            }
            static const u64 short_bytes = [] {
                const char* value = std::getenv("EDEN_NX2_TC_POOL_SHORT_MIB");
                return (value != nullptr ? std::strtoull(value, nullptr, 10) : 1024ULL) << 20;
            }();
            static const u64 critical_bytes = [] {
                const char* value = std::getenv("EDEN_NX2_TC_POOL_CRITICAL_MIB");
                return (value != nullptr ? std::strtoull(value, nullptr, 10) : 256ULL) << 20;
            }();
            static unsigned long long refusals_seen = 0;
            static u64 refusal_frame = 0;
            const auto refusals = ::Eden::Performance::graphics_refusals.load(std::memory_order_relaxed);
            if (refusals != refusals_seen) {
                refusals_seen = refusals;
                refusal_frame = frame_tick;
            }
            const auto free_memory = ::Eden::Performance::graphics_memory_free.load(std::memory_order_relaxed);
            const u64 free_bytes = free_memory ? free_memory() : ~u64{0};
            if ((refusal_frame != 0 && frame_tick - refusal_frame < 120) || free_bytes < critical_bytes) {
                return 2;
            }
            return free_bytes < short_bytes ? 1 : 0;
        }
    };
'''
gc_setup = '''    const auto ReclaimedBytes = [&](const Image& image) {
        u64 bytes = 0;
''' + reclaimed_bytes + '''        return bytes;
    };
''' + (port / 'vulkan_gc_downloads.inc').read_text()
cleanup_begin = gc.index('    const auto Cleanup = [')
cleanup_end = gc.index('](ImageId image_id)', cleanup_begin) + 1
cleanup_capture = gc[cleanup_begin:cleanup_end]
for old, new in [
    (cleanup_capture,
     gc_setup + '\n    const auto Cleanup = [&, this]'),
    ('        if (must_download) {', '        if (must_download && !TakePrefetched(image_id, image)) {'),
    # A refused staging buffer leaves the image where it is: it is not deleted, nothing is lost.
    ('            auto map = runtime.DownloadStagingBuffer(image.unswizzled_size_bytes);\n',
     '            auto staged = [&] {\n'
     '                if constexpr (std::is_same_v<Runtime, Vulkan::TextureCacheRuntime>) {\n'
     '                    return runtime.TryDownloadStagingBuffer(image.unswizzled_size_bytes);\n'
     '                } else {\n'
     '                    return std::optional{runtime.DownloadStagingBuffer(image.unswizzled_size_bytes)};\n'
     '                }\n'
     '            }();\n'
     '            if (!staged) {\n                return false;\n            }\n'
     '            auto& map = *staged;\n'),
    # Time the per-image drain Cleanup still takes for an image the prefetch could not batch.
    # One GPU pipeline wait per eviction, against one wait for a whole batch in the batched path:
    # this is the stall the prefetch exists to remove, so keep it visible.
    ('            image.DownloadMemory(map, copies);\n            runtime.Finish();\n'
     '            SwizzleImage(*gpu_memory, image.gpu_addr, image.info, copies, map.mapped_span, swizzle_data_buffer);',
     '            image.DownloadMemory(map, copies);\n'
     '            {\n'
     '                const ::Eden::Performance::Timer drain_timer(\n'
     '                    ::Eden::Performance::gc_unbatched_drain, 1);\n'
     '                runtime.Finish();\n'
     '            }\n'
     '            SwizzleImage(*gpu_memory, image.gpu_addr, image.info, copies, map.mapped_span, swizzle_data_buffer);'),
    # Copying a dirty image out needs a staging buffer and a scratch buffer of its size, and its bytes
    # then land in guest memory: on a pool the CPU shares, evicting a large one takes more than it gives
    # (Ocarina of Time's 288 MiB depth image: 576 MiB more at that moment, 288 MiB for good).
    ('(!high_priority_mode && must_download)) {',
     '(!DirtyEvictions() && must_download) ||\n            (must_download && image.unswizzled_size_bytes >= 128_MiB)) {'),
    # The pool's shortness as a signal of its own (performance.h texture_usage_from_images), beside
    # what the images add up to: a refusal in the last 120 frames or the largest free block under
    # EDEN_NX2_TC_POOL_CRITICAL_MIB (256) is level 2, collected as past "critical" (images unused
    # for 10 frames go); under EDEN_NX2_TC_POOL_SHORT_MIB (1024, the pool's own "short" mark) level
    # 1, as past "expected" (unused for 25 frames). Run 58 (marks raised to 7000/8000, nothing
    # evicted) held 1058 unused images, 1879 MiB, when the pool refused at 99 s, 20 s after the
    # largest free block fell under 1 GiB: evicting what is unused is what keeps a session alive, so
    # the signal must fire there; images used in the last 10 to 25 frames stay. With the pool
    # figure (EDEN_NX2_TC_POOL_PRESSURE=1) the level is 0 and the marks decide alone, as before.
    ('    bool high_priority_mode = false;\n    bool aggressive_mode = false;\n    bool evict_old_costly = false;\n',
     '    bool high_priority_mode = false;\n    bool aggressive_mode = false;\n    bool evict_old_costly = false;\n'
     + POOL_PRESSURE),
    ('        high_priority_mode = total_used_memory >= expected_memory;\n'
     '        aggressive_mode = allow_aggressive && total_used_memory >= critical_memory;',
     '        const int pressure = PoolPressure();\n'
     '        high_priority_mode = total_used_memory >= expected_memory || pressure >= 1;\n'
     '        aggressive_mode = allow_aggressive && (total_used_memory >= critical_memory || pressure >= 2);'),
    ('        if (aggressive_mode && total_used_memory < critical_memory) {',
     '        if (aggressive_mode && total_used_memory < critical_memory && PoolPressure() < 2) {'),
    ('        } else if (high_priority_mode && total_used_memory < expected_memory) {',
     '        } else if (high_priority_mode && total_used_memory < expected_memory && PoolPressure() < 1) {'),
    ('    if (total_used_memory >= critical_memory) {\n        Configure(true);',
     '    if (total_used_memory >= critical_memory || PoolPressure() >= 2) {\n        Configure(true);'),
    ('    Configure(false);', '    Configure(false);\n    PrefetchDownloads();'),
    ('        Configure(true);', '        Configure(true);\n        PrefetchDownloads();'),
]:
    if gc.count(old) != 1:
        raise RuntimeError(f'Pinned GC anchor changed: {old}')
    gc = gc.replace(old, new)
texture_costs = []
for signature, index in (
    ('RunGarbageCollector()', 13), ('DownloadMemory(DAddr cpu_addr, size_t size)', 14),
    ('PopAsyncFlushes()', 15),
):
    anchor = f'void TextureCache<P>::{signature} {{'
    scope = f'''
    std::optional<Eden::Performance::Timer> cost_timer;
    if constexpr (std::is_same_v<Runtime, Vulkan::TextureCacheRuntime>) {{
        if (Eden::Performance::vulkan_cost_enabled.load(std::memory_order_relaxed))
            cost_timer.emplace(Eden::Performance::vulkan_api[{index}]);
    }}
'''
    if index == 13:
        scope += '''    if constexpr (std::is_same_v<Runtime, Vulkan::TextureCacheRuntime>) {
        if ((cost_timer || Eden::Performance::texture_budget_log.load(std::memory_order_relaxed)) &&
            frame_tick % 300 == 0)
            std::printf("EDEN_VULKAN_TEXTURE_BUDGET frame=%llu usage=%llu expected=%llu critical=%llu "
                        "keep_dirty=%d memory_short=%d\\n",
                        static_cast<unsigned long long>(frame_tick),
                        static_cast<unsigned long long>(total_used_memory),
                        static_cast<unsigned long long>(expected_memory),
                        static_cast<unsigned long long>(critical_memory),
                        int(Eden::Performance::gc_keep_dirty.load(std::memory_order_relaxed)),
                        int(Eden::Performance::graphics_memory_short.load(std::memory_order_relaxed)));
    }
'''
    texture_costs.append((anchor, anchor + scope))
adapt('src/video_core/texture_cache/texture_cache.h',
      'vulkan-cache/video_core/texture_cache/texture_cache.h', [
    ('namespace VideoCommon {', '#include "performance.h"\n#include "common/scope_exit.h"\nnamespace Vulkan { class TextureCacheRuntime; }\n\nnamespace VideoCommon {'),
    ('    // If we can obtain the memory info, use it instead of the estimate.\n    if (runtime.CanReportMemoryUsage()) {',
     "    // The console: the images' own memory (performance.h texture_usage_from_images), the figure\n    // Eden keeps itself when the driver gives none, set right every 300 frames from the images there\n    // are (scaled ones hold their original too, as the collector counts them).\n    const bool from_images = std::is_same_v<Runtime, Vulkan::TextureCacheRuntime> &&\n                             ::Eden::Performance::texture_usage_from_images.load(std::memory_order_relaxed);\n    if (from_images && frame_tick % 300 == 0) {\n        u64 held = 0;\n        for (const auto [image_id, image] : slot_images) {\n            held += ImageMemoryBytes(*image) + (image->HasScaled() ? GetScaledImageSizeBytes(*image) : 0);\n        }\n        total_used_memory = held;\n    }\n    // If we can obtain the memory info, use it instead of the estimate.\n    if (runtime.CanReportMemoryUsage() && !from_images) {"),
    (download_prefix, download_prefix + (port / 'vulkan_download_batch.inc').read_text() + '\n'),
    (gc_original, gc),
    # RADV reports all Vulkan allocations, not just cached textures. A game's
    # measured ~1.9 GiB working set triggered dirty eviction at the old 1.6 GiB
    # threshold despite a 4 GiB budget. Retain 40% headroom; keep the original
    # critical threshold and minimum 1 GiB spacing. OpenGL remains unchanged.
    ('const s64 min_vacancy_expected = (6 * mem_threshold) / 10;',
     'const s64 min_vacancy_expected =\n            ((std::is_same_v<Runtime, Vulkan::TextureCacheRuntime> ? 4 : 6) * mem_threshold) / 10;'),
] + texture_costs)

# The four calls inside bind_geometry and the three inside bind_update. They sit in buffer_cache.h
# rather than in Configure, and both spend their time in SynchronizeBuffer's page walk over the
# memory tracker - once for the index buffer, once per enabled vertex buffer, once per per-stage
# segment, every draw, dirty or not. Which of the seven that is cannot be decided by reading it:
# the texture-buffer loop looked like the cost of bind and measured 2.8%.
adapt('src/video_core/buffer_cache/buffer_cache.h',
      'vulkan-cache/video_core/buffer_cache/buffer_cache.h', [
    ('        if (buffer_bytes >= collect_mark) {\n'
     '            RunGarbageCollector(buffer_bytes >= 2 * collect_mark);\n'
     '        }\n',
     '        const auto pressure = ::Eden::Performance::graphics_memory_probe.load(std::memory_order_relaxed);\n'
     '        const bool memory_short = frame_tick >= 60 && pressure && pressure();\n'
     '        if (buffer_bytes >= collect_mark || memory_short) {\n'
     '            RunGarbageCollector(memory_short || buffer_bytes >= 2 * collect_mark);\n'
     '        }\n'),
    ('++frames % 600 == 0', '++frames % 60 == 0'),
    ('#include "video_core/buffer_cache/buffer_cache_base.h"',
     '#include "video_core/buffer_cache/buffer_cache_base.h"\n#include "performance.h"'),
    ('void BufferCache<P>::BindHostGeometryBuffers(bool is_indexed) {',
     'void BufferCache<P>::BindHostGeometryBuffers(bool is_indexed) {\n'
     '    long long since_geo = ::Eden::Performance::MarkStart();'),
    ('    BindHostVertexBuffers();\n'
     '    BindHostTransformFeedbackBuffers();\n'
     '    if (current_draw_indirect) {\n'
     '        BindHostDrawIndirectBuffers();\n'
     '    }\n'
     '}',
     '    ::Eden::Performance::Mark(::Eden::Performance::geo_index, since_geo);\n'
     '    BindHostVertexBuffers();\n'
     '    ::Eden::Performance::Mark(::Eden::Performance::geo_vertex, since_geo);\n'
     '    BindHostTransformFeedbackBuffers();\n'
     '    ::Eden::Performance::Mark(::Eden::Performance::geo_xf, since_geo);\n'
     '    if (current_draw_indirect) {\n'
     '        BindHostDrawIndirectBuffers();\n'
     '    }\n'
     '    ::Eden::Performance::Mark(::Eden::Performance::geo_indirect, since_geo);\n'
     '}'),
    ('void BufferCache<P>::DoUpdateGraphicsBuffers(bool is_indexed) {',
     'void BufferCache<P>::DoUpdateGraphicsBuffers(bool is_indexed) {\n'
     '    long long since_upd = ::Eden::Performance::MarkStart();'),
    ('        UpdateTransformFeedbackBuffers();\n'
     '        for (size_t stage = 0; stage < NUM_STAGES; ++stage) {\n'
     '            UpdateUniformBuffers(stage);\n'
     '            UpdateStorageBuffers(stage);\n'
     '            UpdateTextureBuffers(stage);\n'
     '        }\n'
     '        if (current_draw_indirect) {\n'
     '            UpdateDrawIndirect();\n'
     '        }',
     '        UpdateTransformFeedbackBuffers();\n'
     '        ::Eden::Performance::Mark(::Eden::Performance::upd_state, since_upd);\n'
     '        for (size_t stage = 0; stage < NUM_STAGES; ++stage) {\n'
     '            UpdateUniformBuffers(stage);\n'
     '            UpdateStorageBuffers(stage);\n'
     '            UpdateTextureBuffers(stage);\n'
     '        }\n'
     '        ::Eden::Performance::Mark(::Eden::Performance::upd_stage, since_upd);\n'
     '        if (current_draw_indirect) {\n'
     '            UpdateDrawIndirect();\n'
     '        }\n'
     '        ::Eden::Performance::Mark(::Eden::Performance::upd_indirect, since_upd);'),
])

# These costs sit outside vkCreateGraphicsPipelines: guest shader translation,
# worker readiness, and background serialization can block different threads.
# The six phases inside Configure. draw_config is all of Configure but SetEngine's two pointer
# assignments, and ConfigureImpl is ~210 lines; without these the 2.78 us a draw spends
# configuring arrives as one number with nothing to aim at. One running clock, marked at the
# boundaries the source already has.
configure_phases = [
    ('bool GraphicsPipeline::ConfigureImpl(bool is_indexed) {',
     'bool GraphicsPipeline::ConfigureImpl(bool is_indexed) {\n'
     '    long long since = ::Eden::Performance::MarkStart();'),
    ('    buffer_cache.SetUniformBuffersState(enabled_uniform_buffer_masks, &uniform_buffer_sizes);',
     '    buffer_cache.SetUniformBuffersState(enabled_uniform_buffer_masks, &uniform_buffer_sizes);\n'
     '    // Sampler and image tables synced, uniform-buffer state set.\n'
     '    ::Eden::Performance::Mark(::Eden::Performance::cfg_sync, since);'),
    ('        texture_cache.FillImageViews(std::span(views.data(), views.size()), false,\n'
     '                                     Spec::has_images);\n    }\n',
     '        texture_cache.FillImageViews(std::span(views.data(), views.size()), false,\n'
     '                                     Spec::has_images);\n    }\n'
     '    // Every texture handle read out of guest memory, every sampler and view filled.\n'
     '    ::Eden::Performance::Mark(::Eden::Performance::cfg_read, since);\n'
     '    long long since_bind = ::Eden::Performance::MarkStart();'),
    ('    if (regs.transform_feedback_enabled != 0) {',
     '    // The texture-buffer loop over every enabled stage.\n'
     '    ::Eden::Performance::Mark(::Eden::Performance::bind_tbo, since_bind);\n'
     '    if (regs.transform_feedback_enabled != 0) {'),
    ('    buffer_cache.UpdateGraphicsBuffers(is_indexed);',
     '    buffer_cache.UpdateGraphicsBuffers(is_indexed);\n'
     '    // Index, vertex, transform-feedback and per-stage buffer updates done.\n'
     '    ::Eden::Performance::Mark(::Eden::Performance::bind_update, since_bind);'),
    ('    buffer_cache.BindHostGeometryBuffers(is_indexed);',
     '    buffer_cache.BindHostGeometryBuffers(is_indexed);\n'
     '    // Storage, texture and geometry buffers bound.\n'
     '    ::Eden::Performance::Mark(::Eden::Performance::bind_geometry, since_bind);\n'
     '    ::Eden::Performance::Mark(::Eden::Performance::cfg_bind, since);'),
    ('    texture_cache.UpdateRenderTargets(false);',
     '    // Descriptor queue acquired, image descriptors pushed, uploads barriered.\n'
     '    ::Eden::Performance::Mark(::Eden::Performance::cfg_desc, since);\n'
     '    texture_cache.UpdateRenderTargets(false);'),
    ('    if (IsBuilt() && !pipeline) {',
     '    // Render targets current, feedback loops checked.\n'
     '    ::Eden::Performance::Mark(::Eden::Performance::cfg_rt, since);\n'
     '    if (IsBuilt() && !pipeline) {'),
    ('    return ConfigureDraw(rescaling, render_area);',
     '    // Compare the descriptor payload against last draw\'s and record the bindings.\n'
     '    const bool configured = ConfigureDraw(rescaling, render_area);\n'
     '    ::Eden::Performance::Mark(::Eden::Performance::cfg_draw, since);\n'
     '    return configured;'),
]
for name in ('vk_graphics_pipeline', 'vk_compute_pipeline'):
    anchors = [
        ('#include <algorithm>', '#include <algorithm>\n#include "performance.h"'),
        ('            std::unique_lock lock{build_mutex};',
         '            auto timer = Eden::Performance::VulkanTimer(16);\n            std::unique_lock lock{build_mutex};'),
    ]
    if name == 'vk_graphics_pipeline':
        anchors += configure_phases
    adapt(f'src/video_core/renderer_vulkan/{name}.cpp', name + '_cost.cpp', anchors)
shader_costs = [
    ('#include <algorithm>', '#include <algorithm>\n#include <cstdio>\n#include <unordered_set>\n'
     '#include "performance.h"\n#include "dev_vulkan.h"\n#include "diagnostics.h"'),
    # Development crash probes: report each pipeline's first use (stage hashes) to klog,
    # which survives a GPU fault that kills the title.
    ('    current_pipeline->MarkUsed(frame_count);\n    return BuiltPipeline(current_pipeline);\n}',
     '    current_pipeline->MarkUsed(frame_count);\n'
     '    if (::Eden::DevVulkan::trace_pipelines) {\n'
     '        static std::unordered_set<const void*> traced;\n'
     '        if (traced.insert(current_pipeline).second) {\n'
     '            const auto& h = graphics_key.unique_hashes;\n'
     '            char text[256];\n'
     '            std::snprintf(text, sizeof(text), "G%zu p=%p new=%d va=%llx vb=%llx tc=%llx te=%llx gs=%llx fs=%llx",\n'
     '                          traced.size(), (const void*)current_pipeline, int(is_new), (unsigned long long)h[0], (unsigned long long)h[1],\n'
     '                          (unsigned long long)h[2], (unsigned long long)h[3], (unsigned long long)h[4],\n'
     '                          (unsigned long long)h[5]);\n'
     '            ::Eden::Report("pipeline", text);\n'
     '            LOG_INFO(Render_Vulkan, "EDEN_PIPELINE_FIRST_USE {}", text);\n'
     '        }\n'
     '    }\n'
     '    return BuiltPipeline(current_pipeline);\n}'),
    ('    if (!is_new) {\n        return pipeline.get();\n    }\n    pipeline = CreateComputePipeline(key, shader);\n    return pipeline.get();\n}',
     '    if (is_new) {\n'
     '        pipeline = CreateComputePipeline(key, shader);\n'
     '    }\n'
     '    if (::Eden::DevVulkan::trace_pipelines && pipeline) {\n'
     '        static std::unordered_set<const void*> traced;\n'
     '        if (traced.insert(pipeline.get()).second) {\n'
     '            char text[160];\n'
     '            std::snprintf(text, sizeof(text), "C%zu p=%p new=%d cs=%llx shared=%u wg=%u,%u,%u", traced.size(),\n'
     '                          (const void*)pipeline.get(), int(is_new), (unsigned long long)key.unique_hash, unsigned(key.shared_memory_size),\n'
     '                          unsigned(key.workgroup_size[0]), unsigned(key.workgroup_size[1]),\n'
     '                          unsigned(key.workgroup_size[2]));\n'
     '            ::Eden::Report("pipeline", text);\n'
     '            LOG_INFO(Render_Vulkan, "EDEN_PIPELINE_FIRST_USE {}", text);\n'
     '        }\n'
     '    }\n'
     '    return pipeline.get();\n}'),
    ('std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline() {',
     'std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline() {\n    auto timer = Eden::Performance::VulkanTimer(17);'),
    ('    const ComputePipelineCacheKey& key, const ShaderInfo* shader) {',
     '    const ComputePipelineCacheKey& key, const ShaderInfo* shader) {\n    auto timer = Eden::Performance::VulkanTimer(17);'),
    ('                                                 u32 cache_version) {\n    // Written whole beside',
     '                                                 u32 cache_version) {\n    auto timer = Eden::Performance::VulkanTimer(18);\n    // Written whole beside'),
    ('        device.SupportsTransformFeedbackProvokingVertexPreservation();\n}\n\nnamespace {',
     '        device.SupportsTransformFeedbackProvokingVertexPreservation();\n'
     '    if (Eden::Performance::vulkan_cost_enabled.load(std::memory_order_relaxed))\n'
     '        std::printf("EDEN_VULKAN_DYNAMIC blend=%u enables=%u vertex=%u\\n",\n'
     '                    unsigned(dynamic_features.has_extended_dynamic_state_3_blend),\n'
     '                    unsigned(dynamic_features.has_extended_dynamic_state_3_enables),\n'
     '                    unsigned(dynamic_features.has_dynamic_vertex_input));\n'
     '}\n\nnamespace {'),
]
# Time both graphics and compute preparation, without modifying the compiler or
# reading guest state on another thread. These intervals overlap shader_prepare.
shader_source = (source / 'src/video_core/renderer_vulkan/vk_pipeline_cache.cpp').read_text()
shader_source = shader_source.replace('    auto it{graphics_cache.find(graphics_key)};\n    if (it == graphics_cache.end()) {',
                                      '    auto it{graphics_cache.find(graphics_key)};\n    const bool is_new = it == graphics_cache.end();\n    if (is_new) {')
for expression, index, count in (
    ('main_pools.ReleaseContents()', 19, 2),
    ('TranslateProgram(pools.inst, pools.block, env, cfg, host_info)', 21, 3),
    # 2 since 0509 (EDEN_UP_STAGE_CACHE): its verify path emits the stage again.
    ('EmitSPIRV(profile, runtime_info, program, binding)', 22, 2),
    ('EmitSPIRV(profile, program)', 22, 1),
    ('BuildShader(device, code)', 23, 2),
):
    if shader_source.count(expression) != count:
        raise RuntimeError(f'Shader cost anchor changed: {expression}')
    shader_source = shader_source.replace(expression,
        f'([&] {{ auto cost = Eden::Performance::VulkanTimer({index}); '
        f'return {expression}; }}())')
for statement in (
    'Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);',
    'Shader::Maxwell::Flow::CFG cfg{env, pools.flow_block, env.StartAddress()};',
):
    # graphics exception dump too; 3 since 0509 (EDEN_UP_STAGE_CACHE verifies a stage again)
    expected = 3 if 'cfg_offset' in statement else 1
    if shader_source.count(statement) != expected:
        raise RuntimeError(f'Shader CFG anchor changed: {statement}')
    shader_source = shader_source.replace(statement,
        'auto cfg_cost = Eden::Performance::VulkanTimer(20);\n        ' + statement +
        '\n        cfg_cost.reset();')
for old, new in shader_costs:
    if old == '}\n\nPipelineCache::~PipelineCache()' and old not in shader_source:
        boundary = '}\n\nnamespace {\n// <title>/vulkan_used.bin'
        new = new.replace(old, boundary)
        old = boundary
    if old.startswith('    current_pipeline = pipeline.get();') and old not in shader_source:
        mark_used = '    current_pipeline->MarkUsed(frame_count);\n'
        old = old.replace('    return BuiltPipeline', mark_used + '    return BuiltPipeline')
        new = new.replace('    current_pipeline = pipeline.get();\n',
                          '    current_pipeline = pipeline.get();\n' + mark_used)
    if old.startswith('    if (!is_new) {\n        return pipeline.get();\n    }\n') and old not in shader_source:
        # 0345-up-null-pointer-slots puts its pointer-slot rebuild inside the !is_new block: keep it,
        # and let that block fall through to the trace below instead of returning early.
        tail = '\n        return pipeline.get();\n    }\n    pipeline = CreateComputePipeline(key, shader);\n    return pipeline.get();\n}'
        if shader_source.count(tail) == 1:
            old, new = tail, '\n    }\n' + new
    if old == '    const ComputePipelineCacheKey& key, const ShaderInfo* shader) {' and old not in shader_source:
        # 0345-up-null-pointer-slots adds a `previous` parameter.
        wider = '    const ComputePipelineCacheKey& key, const ShaderInfo* shader,\n    const ComputePipeline* previous) {'
        if shader_source.count(wider) == 1:
            new = new.replace(old, wider)
            old = wider
    if 'Shader::TensorRejected' in old and 'Shader::TensorRejected' not in shader_source:
        old = ('    if (!is_new) {\n        return pipeline.get();\n    }\n'
               '    pipeline = CreateComputePipeline(key, shader);\n    return pipeline.get();\n}')
        trace = new[new.index('    if (::Eden::DevVulkan::trace_pipelines && pipeline)'):]
        new = ('    if (is_new) {\n        pipeline = CreateComputePipeline(key, shader);\n    }\n' + trace)
    if shader_source.count(old) != 1:
        raise RuntimeError(f'Shader pipeline anchor changed: {old[:60]}')
    shader_source = shader_source.replace(old, new)
shader_output = output / 'vk_pipeline_cache_cost.cpp'
if not shader_output.exists() or shader_output.read_text() != shader_source:
    shader_output.write_text(shader_source)

# Scope only the existing synchronous operations; never alter their ordering.
# Report these caller costs separately from the overlapping API totals.
import re
# Development GPU fault probes (dev_vulkan.h): multi-range storage buffers are
# the one path that binds sparse ranges of the console's device memory.
probes = {
    'vk_multi_range_buffer': [
        ('#include <algorithm>', '#include <algorithm>\n#include <cstdio>\n#include "dev_vulkan.h"\n#include "diagnostics.h"'),
        ('    if (bind_result != VK_SUCCESS) {\n        return SparseBuffer{};\n    }',
         '    if (bind_result != VK_SUCCESS) {\n'
         '        if (::Eden::DevVulkan::trace_pipelines) ::Eden::Report("multirange", "sparse bind failed");\n'
         '        return SparseBuffer{};\n    }'),
        ('    retired.push_back(Retired{',
         '    if (::Eden::DevVulkan::trace_pipelines) {\n'
         '        static unsigned retires = 0;\n'
         '        if (++retires <= 64) {\n'
         '            char text[96];\n'
         '            std::snprintf(text, sizeof(text), "retire #%u sparse=%d tick=%llu", retires,\n'
         '                          int(bool(entry.sparse_handle)), (unsigned long long)scheduler.CurrentTick());\n'
         '            ::Eden::Report("multirange", text);\n'
         '        }\n'
         '    }\n'
         '    retired.push_back(Retired{'),
        ('    entries.emplace(key, std::move(entry));',
         '    if (::Eden::DevVulkan::trace_pipelines) {\n'
         '        static unsigned created = 0;\n'
         '        if (++created <= 256) {\n'
         '            char text[192];\n'
         '            std::snprintf(text, sizeof(text), "new #%u key=%llx total=%llu sources=%zu sparse=%d addr=%llx",\n'
         '                          created, (unsigned long long)key, (unsigned long long)total, sources.size(),\n'
         '                          int(bool(entry.sparse_handle)), (unsigned long long)entry.address);\n'
         '            ::Eden::Report("multirange", text);\n'
         '        }\n'
         '    }\n'
         '    entries.emplace(key, std::move(entry));'),
    ],
    'vk_buffer_cache': [
        ('#include <algorithm>', '#include <algorithm>\n#include "dev_vulkan.h"\n#include "submission_origin.h"'),
        ('void BufferCacheRuntime::Finish() {',
         'void BufferCacheRuntime::Finish() {\n'
         '    ::Eden::VulkanProbe::GpuSubmitOriginScope origin{__builtin_return_address(0), true};'),
        ('bool BufferCacheRuntime::BindMultiRangeStorageBuffer(u64 key, bool is_written) {\n',
         'bool BufferCacheRuntime::BindMultiRangeStorageBuffer(u64 key, bool is_written) {\n'
         '    if (::Eden::DevVulkan::disable_multi_range) {\n        return false;\n    }\n'),
    ],
    # With custom border colours off (the console default), report which clamp-to-border
    # samplers ask for a colour the three standard ones cannot give exactly.
    'vk_texture_cache': [
        ('void TextureCacheRuntime::Finish() {',
         'void TextureCacheRuntime::Finish() {\n'
         '    ::Eden::VulkanProbe::GpuSubmitOriginScope origin{__builtin_return_address(0), true};'),
        ('void TextureCacheRuntime::TranscodeAstcToBcn(\n'
         '    Image& image, const StagingBufferRef& map,\n'
         '    std::span<const VideoCommon::SwizzleParameters> swizzles) {\n',
         'void TextureCacheRuntime::TranscodeAstcToBcn(\n'
         '    Image& image, const StagingBufferRef& map,\n'
         '    std::span<const VideoCommon::SwizzleParameters> swizzles) {\n'
         '    ::Eden::Performance::Timer mkw_timer(::Eden::Performance::mkw_transcode);\n'),
        # Decode images can otherwise survive a long loading burst until TickFrame. At this
        # boundary every use of the input and output staging buffers has been recorded, so a
        # wait is safe (unlike a wait inside an arbitrary staging-buffer allocation).
        ('        .image = memory_allocator.CreateImage(VkImageCreateInfo{\n            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,',
         '        .image = memory_allocator.CreateTemporaryImage(VkImageCreateInfo{\n            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,'),
        # A full GPU wait per texture was 5% of the split's back thread at Mario Kart World's
        # title (one streamed texture a frame, the pool always short): the decode images are now
        # reclaimed once the ones in flight exceed EDEN_PS5_TRANSCODE_PENDING_MIB (64), waiting
        # for the oldest only. EDEN_PS5_TRANSCODE_FINISH=1 restores the wait after each.
        ('    pending_transcodes.push_back(std::move(pending));\n',
         '    pending_transcodes.push_back(std::move(pending));\n'
         '    const auto memory_short = ::Eden::Performance::graphics_memory_probe.load(std::memory_order_relaxed);\n'
         '    if (memory_short && memory_short()) {\n'
         '        static const bool finish_each = [] {\n'
         '            const char* value = std::getenv("EDEN_PS5_TRANSCODE_FINISH");\n'
         '            return value != nullptr && value[0] == \'1\';\n'
         '        }();\n'
         '        static const u64 pending_limit = [] {\n'
         '            const char* value = std::getenv("EDEN_PS5_TRANSCODE_PENDING_MIB");\n'
         '            return (value != nullptr ? std::strtoull(value, nullptr, 10) : u64{64}) << 20;\n'
         '        }();\n'
         '        ::Eden::Performance::Timer finish_timer(::Eden::Performance::mkw_transcode_finish);\n'
         '        if (finish_each) {\n'
         '            scheduler.Finish();\n'
         '        }\n'
         '        while (!pending_transcodes.empty() && scheduler.IsFree(pending_transcodes.front().tick)) {\n'
         '            pending_transcodes.pop_front();\n'
         '        }\n'
         '        const auto decode_bytes = [](const PendingTranscode& entry) {\n'
         '            return u64{entry.width} * entry.height * 4 * entry.layers * (entry.levels > 1 ? 4 : 3) / 3;\n'
         '        };\n'
         '        u64 pending_bytes = 0;\n'
         '        for (const auto& entry : pending_transcodes) {\n'
         '            pending_bytes += decode_bytes(entry);\n'
         '        }\n'
         '        while (pending_bytes > pending_limit && pending_transcodes.size() > 1) {\n'
         '            scheduler.Wait(pending_transcodes.front().tick);\n'
         '            pending_bytes -= decode_bytes(pending_transcodes.front());\n'
         '            pending_transcodes.pop_front();\n'
         '        }\n'
         '        staging_buffer_pool.ReleaseCompleted();\n'
         '    }\n'),
        ('#include <algorithm>', '#include <algorithm>\n#include <cstdio>\n#include "dev_vulkan.h"\n#include "diagnostics.h"\n#include "submission_origin.h"'),
        ('    has_custom_border_colors = samples_border && device.IsCustomBorderColorUsable();',
         '    if (::Eden::DevVulkan::trace_pipelines && samples_border) {\n'
         '        static std::vector<std::array<float, 4>> reported;\n'
         '        const auto report = [](const std::array<float, 4>& c, const char* kind) {\n'
         '            if (c == std::array<float, 4>{0, 0, 0, 0} || c == std::array<float, 4>{0, 0, 0, 1} ||\n'
         '                c == std::array<float, 4>{1, 1, 1, 1} || reported.size() >= 32 ||\n'
         '                std::find(reported.begin(), reported.end(), c) != reported.end()) return;\n'
         '            reported.push_back(c);\n'
         '            char text[128];\n'
         '            std::snprintf(text, sizeof(text), "non-standard %s border %g %g %g %g", kind, c[0], c[1],\n'
         '                          c[2], c[3]);\n'
         '            ::Eden::Report("sampler", text);\n'
         '        };\n'
         '        report(border_color, "linear");\n'
         '        if (tsc.srgb_conversion) report(srgb_border_color, "srgb");\n'
         '    }\n'
         '    has_custom_border_colors = samples_border && device.IsCustomBorderColorUsable();'),
    ],
}
for name, index, expected in (
    ('vk_fence_manager', 6, 1), ('vk_buffer_cache', 7, 6),  # three diagnostic mapped-readback probe waits
    ('vk_texture_cache', 8, 1), ('vk_swapchain', 9, 6),
    ('vk_descriptor_buffer', 10, 2), ('vk_multi_range_buffer', 11, 1),
    ('vk_compute_pass', 12, 1),
):
    relative = f'src/video_core/renderer_vulkan/{name}.cpp'
    destination = 'vulkan_swapchain.cpp' if name == 'vk_swapchain' else name + '_cost.cpp'
    # Preserve the earlier swapchain error-recovery adaptation.
    path = output / destination
    text = path.read_text() if name == 'vk_swapchain' else (source / relative).read_text()
    pattern = r'(?m)^( +)(scheduler\.(?:Wait|Finish)\([^\n]*\);)$'
    matches = list(re.finditer(pattern, text))
    if len(matches) != expected:
        print(f'[ANCHOR-SKIPPED] wait sites of {name}: {len(matches)}, expected {expected}')
    text = re.sub(pattern, lambda m: m[1] + '{ auto timer = Eden::Performance::VulkanTimer('
                  + str(index) + '); ' + m[2] + ' }', text)
    if name == 'vk_compute_pass':
        # ASTC decode has no CPU consumer. Its existing image barrier orders
        # later shaders, and staging/descriptor reuse is protected by ticks.
        text = text.replace('VulkanTimer(12); scheduler.Finish();',
                            'VulkanTimer(12); scheduler.Flush();')
    for probe_old, probe_new in probes.get(name, []):
        if text.count(probe_old) != 1:
            print(f'[ANCHOR-SKIPPED] probe of {name}: {probe_old[:90]!r}')
            continue
        text = text.replace(probe_old, probe_new)
    text = '#include "performance.h"\n' + text
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
# Eden loads these shaders when the renderer starts, whether frames are to be generated or not:
# the reasons are only written when the setting is on, or every session without the user's
# Lossless.dll would say so.
adapt('src/video_core/renderer_vulkan/present/lsfg_shaders.cpp', 'vulkan_lsfg_shaders.cpp', [
    ('#include "video_core/vulkan_common/vulkan_device.h"',
     '#include <string>\n#include "common/settings.h"\n#include "video_core/vulkan_common/vulkan_device.h"\n#include "diagnostics.h"'),
    ("""    if (!device.IsVulkanMemoryModelSupported() || !device.HasNullDescriptor() ||
        !device.IsFloat16Supported()) {
        return;
    }""",
     """    const bool wanted = Settings::values.frame_gen.GetValue();
    if (!device.IsVulkanMemoryModelSupported() || !device.HasNullDescriptor() ||
        !device.IsFloat16Supported()) {
        if (wanted)
            Eden::Report("framegen",
                         ("device features: vulkanMemoryModel=" +
                          std::to_string(device.IsVulkanMemoryModelSupported()) + " nullDescriptor=" +
                          std::to_string(device.HasNullDescriptor()) + " float16=" +
                          std::to_string(device.IsFloat16Supported()) +
                          " -- all are required, and dev-settings.txt null_descriptor=off turns "
                          "the second one off").c_str());
        return;
    }"""),
    ("""    VideoCore::FrameGen::ShaderModules code;
    if (VideoCore::FrameGen::LoadShaderModules(code) != VideoCore::FrameGen::LosslessStatus::Ok) {
        return;
    }""",
     """    VideoCore::FrameGen::ShaderModules code;
    const VideoCore::FrameGen::LosslessStatus status = VideoCore::FrameGen::LoadShaderModules(code);
    if (status != VideoCore::FrameGen::LosslessStatus::Ok) {
        if (wanted)
            Eden::Report("framegen",
                         ("Lossless.dll " + VideoCore::FrameGen::GetLosslessDllPath().string() +
                          " status=" + std::to_string(static_cast<u32>(status)) +
                          " (0 ok, 1 not installed, 2 unreadable, 3 not a PE, 4 no shaders in it, "
                          "5 translation failed, 6 cache unusable)").c_str());
        return;
    }"""),
    ("""    valid = true;""",
     """    valid = true;
    if (wanted)
        Eden::Report("framegen", ("shaders ready: " + std::to_string(code.size()) + " modules").c_str());"""),
])
# Two members, so that "frames are being generated" and "this image cannot take them" are said once
# per session rather than once per process or once per frame: this object is built per renderer,
# and the renderer is built per game session.
adapt('src/video_core/renderer_vulkan/present/frame_gen.h',
      'include/video_core/renderer_vulkan/present/frame_gen.h', [
    ('    bool dumped{};', '    bool dumped{};\n    bool generation_reported{};\n    bool storage_reported{};'),
])
adapt('src/video_core/renderer_vulkan/present/frame_gen.cpp', 'vulkan_frame_gen.cpp', [
    ('#include "video_core/vulkan_common/vulkan_device.h"',
     '#include "video_core/vulkan_common/vulkan_device.h"\n#include "diagnostics.h"'),
    ("""    if (!shaders->IsValid() && Settings::values.frame_gen_type.GetValue() != Settings::FrameGenType::Fsr3) {
        unavailable = true;
        return;
    }""",
     """    if (!shaders->IsValid() && Settings::values.frame_gen_type.GetValue() != Settings::FrameGenType::Fsr3) {
        Eden::Report("framegen", "LsfgShaders is not valid; see the lines above");
        unavailable = true;
        return;
    }"""),
    ("""    if (!frame->storage_view) {
        warm_streak = 0;
        return;
    }""",
     """    if (!frame->storage_view) {
        if (!storage_reported) {
            storage_reported = true;
            Eden::Report("framegen",
                         "the frame image has no storage view: the swapchain format lacks "
                         "VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT, so frames cannot be generated");
        }
        warm_streak = 0;
        return;
    }"""),
    # The gates above only say why it is not running; this says that it is.
    ("""    generated = warm && warm_streak >= LSFG_RECURRENCE_FRAMES && plan.generations > 0;""",
     """    generated = warm && warm_streak >= LSFG_RECURRENCE_FRAMES && plan.generations > 0;
    if (generated && !generation_reported) {
        generation_reported = true;
        Eden::Report("framegen",
                     ("generating " + std::to_string(plan.generations) +
                      " extra frame(s) per guest frame").c_str());
    }"""),
])
# Split the dispatch into its three phases (Tegra::Control::Scheduler::Push): the channel lock,
# queueing the submit list, and executing the guest's methods. The dispatch timer wraps the whole
# call from gpu_thread.cpp, so these say which of the three it is spent in.
adapt('src/video_core/control/scheduler.cpp', 'control_scheduler.cpp', [
    ('#include <memory>', '#include <memory>\n#include "performance.h"'),
    ('        std::unique_lock lk(scheduling_guard);\n'
     '        auto it = channels.find(channel);\n'
     '        ASSERT(it != channels.end());\n'
     '        channel_state = it->second;\n'
     '    }\n'
     '    BindChannel(gpu, channel_state->bind_id);\n'
     '    // Process commands outside the lock to reduce contention.\n'
     '    // Multiple channels can prepare their commands in parallel.\n'
     '    channel_state->payload->dma_pusher.Push(std::move(entries));\n'
     '    channel_state->payload->dma_pusher.DispatchCalls();',
     '        const ::Eden::Performance::Timer sched_lock_timer(::Eden::Performance::gpu_sched_lock);\n'
     '        std::unique_lock lk(scheduling_guard);\n'
     '        auto it = channels.find(channel);\n'
     '        ASSERT(it != channels.end());\n'
     '        channel_state = it->second;\n'
     '    }\n'
     '    BindChannel(gpu, channel_state->bind_id);\n'
     '    // Process commands outside the lock to reduce contention.\n'
     '    // Multiple channels can prepare their commands in parallel.\n'
     '    {\n'
     '        const ::Eden::Performance::Timer dma_push_timer(::Eden::Performance::gpu_dma_push);\n'
     '        channel_state->payload->dma_pusher.Push(std::move(entries));\n'
     '    }\n'
     '    {\n'
     '        const ::Eden::Performance::Timer dma_dispatch_timer(::Eden::Performance::gpu_dma_dispatch);\n'
     '        channel_state->payload->dma_pusher.DispatchCalls();\n'
     '    }'),
])
shader_header = '#pragma once\n#include <cstdint>\n'
for stage in ('vert', 'frag'):
    binary = output / f'vulkan_hud.{stage}.spv'
    subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.0',
                    '-o', str(binary), str(port / f'vulkan_hud.{stage}')], check=True)
    subprocess.run(['spirv-val', '--target-env', 'vulkan1.0', str(binary)], check=True)
    data = binary.read_bytes()
    words = struct.unpack('<' + 'I' * (len(data) // 4), data)
    shader_header += f'inline constexpr uint32_t EDEN_HUD_{stage.upper()}_SPV[] = {{\n'
    shader_header += ','.join(hex(word) for word in words) + '\n};\n'
path = output / 'vulkan_hud_shaders.h'
if not path.exists() or path.read_text() != shader_header:
    path.write_text(shader_header)
if broken:
    print(f"{len(broken)} anchor(s) of the Eden source changed; adapt them in tools/prepare-vulkan-port.py:", file=sys.stderr)
    for line in broken:
        print("  " + line, file=sys.stderr)
    sys.exit(1)
