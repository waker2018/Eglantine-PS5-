<p align="center">
  <img src="sce_sys/icon0.png" width="128" alt="EglantinePS5 icon">
</p>

<h1 align="center">EglantinePS5 · NUP build</h1>

<p align="center">
  Run Nintendo Switch 1 and <strong>Switch 2</strong> games on a jailbroken PlayStation 5, from folders or single <code>.nup</code> files.<br>
  Ejecuta juegos de Nintendo Switch 1 y <strong>Switch 2</strong> en una PS5 con jailbreak, desde carpetas o ficheros <code>.nup</code>.
</p>

<p align="center">
  <a href="#english">English</a> · <a href="#español">Español</a> · <a href="../../releases">Releases / Descargas</a>
</p>

> [!WARNING]
> Unofficial, experimental software. It is not affiliated with Sony, Nintendo, the Eglantine team or the Eden team. It contains **no** games, keys or firmware: you must bring your own legally dumped files.
> Software no oficial y experimental. Sin relación con Sony, Nintendo ni los equipos de Eglantine o Eden. **No** incluye juegos, claves ni firmware: tienes que aportar los tuyos, volcados legalmente.

---

# English

## What this is

A PS5 homebrew build that stacks three layers, always updated in this order:

| # | Layer | Source | What we do with it |
|---|---|---|---|
| 1 | **Emulator core** | [Djihads80/Eglantine](https://github.com/Djihads80/Eglantine), branch `nx2` (a Switch 2 capable fork of Eden) | Pinned as the `Eglantine/` submodule at its **latest** `nx2` commit. Never forked: our changes are small patches. |
| 2 | **PS5 frontend** | [heydemoura/EglantinePS5](https://github.com/heydemoura/EglantinePS5) (launcher, Vulkan/RADV port, packaging), built on BlackBearReloaded's ProsperoEden | This repository's full history is that project's; we rebase on its `main` and keep our changes on top. |
| 3 | **NUP support** | `.nup` is Eglantine's single-file package (base game + update + DLC) | The library lists `.nup` files, runs them, and can **pack a game folder into a NUP** on the console. |

Current core pin: **Eglantine `0ca2032`** (`nx2`, `build-219` + 35 commits, 2026-11-05).

## Status (honest version)

| Item | State |
|---|---|
| Console tested | PS5 firmware **13.60**, jailbroken, launched with ShadowMountPlus; other users run it on **11.40** and **11.60**. |
| v1.219.029 | Built from v1.219.028 (same core) with one fix: Ocarina of Time no longer asks for a 1 GiB buffer when Kokiri Forest loads, which closed the game there (issue #21). **Compiled, not yet tested on a console.** |
| v1.219.028 | Built 2026-11-08 from v1.219.020 (same core) with one fix: memory given back to the system is reserved in one step (a race that crashed an experimental build). |
| Memory and CPU experiments (builds 021-027, not released) | Swapping the game's memory to the SSD and mimalloc as the app heap did not help; drawing task and mesh shaders through compute did. Measurements and conclusions: [docs/MEMORY-CPU-EXPERIMENTS.md](docs/MEMORY-CPU-EXPERIMENTS.md). |
| Donkey Kong Bananza (Switch 2, `.nup` with update 3.2.0 + DLC) | Runs on the current build (core `3081c8e`) with its update and DLC: **Docked, 1x resolution, 10-25 fps depending on the zone, about 15 in large open areas, 30 fps in cutscenes**. A new zone can show purple the first time it is entered; in one test the game's own "reset terrain" option cleared it. |
| Handheld mode, 0.5x resolution | Not yet confirmed. Earlier failures were caused by a stale shader cache (see below), so they have to be retested. |
| Update and DLC inside a NUP | Both load, and the Library shows them. Checked with Bananza: the game mounts its DLC (log: `NUP add-on 040074A01BF13001 mounted`) and its DLC menu shows 3.2.0 and DLC 1.0.0. The DLC's levels have not been played through yet. |
| Super Mario Bros. Wonder NUP ("Switch 2 Edition + DLC") | The tested file runs as the Switch 1 game with update 1.2.0, and its DLC stays unused: the executables in that file are the Switch 1 build, and the DLC holds Switch 2 data. The Library still lists it. |
| Mario Kart World (Switch 2) | With **Faster GPU emulation** on for the game, v1.219.018 reaches the race at 25-43 fps (0.5x resolution, handheld); with it off, gameplay runs at about 25 of its 30 fps (v1.219.016). Task and mesh draws are off because they hang the PS5's GPU. Every run so far ends after one to three minutes: out of memory (in a race the game itself holds about 5.9 GB), or once a freeze with the GPU thread stopped. The race crash was a single 352 MB request for four 85 MB terrain layers; v1.219.019 uploads those in groups of up to 128 MB, which worked, and then the JIT found no 16 MB block for its code cache while the game's 512 MB memory reserve sat unused. v1.219.020 takes the JIT's memory in 2 MB blocks, lends part of that reserve to the JIT and to a refused graphics allocation (half of it always stays for the game), and gives idle upload buffers back while memory is short (not run yet). Restart the app before each try. |
| The Legend of Zelda: Ocarina of Time (Switch 2) | Plays at about 18 fps on v1.219.015 (one tester's console): its movies and the game show. Its upscaling networks take much of the GPU's time. |
| Donkey Kong Bananza on v1.219.014 | Crashed when many objects broke at once; fixed since v1.219.015. A run on v1.219.016 played at 12-15 fps without that crash. |
| Other Switch 2 titles | Some fail early. Nothing is promised beyond what is listed here. |
| Frame generation (Lossless Scaling) | Compiled in, but pointless at a 20 fps base. |

### Known issue that is fixed here: "stuck at 98% on every launch after the first"

On the console, the Vulkan driver hung a Switch 2 game when it started from the pipeline cache saved by an earlier run. This build neither loads nor saves that global cache (`tools/eglantine-patches/0010-…`); per-game caches still work. If you ever see a hang at 98%, delete `/data/eglantineps5/user/cache/shader/` and tell us (attach `eden_log.txt`).

## Local changes on top of v1.219.029 (not released)

From one console's logs of Ocarina of Time on v1.219.029:

- **Always on Vulkan** (`headless/main.cpp`): on OpenGL the game filled the app's 6 GiB heap. The first run ran out in game (`std::bad_alloc`); every later run ran out while reading the game's OpenGL shader cache and crashed at the same address. A game setting of OpenGL is now overridden for this title, and the launch log says `OpenGL is not used for this game: Vulkan instead`.
- **No task and mesh draws** (`tools/eglantine-patches/0083-…`): with them, every Vulkan run froze about 4 s after the intro movie, with the GPU's back thread inside one command (the same 5838 words left each time). With the testing switch "Task and mesh draws: Off", two runs played on into Kokiri Forest at 16-27 fps. "Task and mesh draws: On" draws them again.
- **ASTC arrays re-compressed in batches** (`0084`, `0085`, from branch `experiment/memoria-cpu`): both of those runs then ended with `VK_ERROR_OUT_OF_DEVICE_MEMORY` at the game's 1365x1365, 90-layer texture array. In the experimental builds the game still ran out of memory later (at 584 s on 01.219.027), so this moves the end, it does not remove it.
- Not tested on a console yet.

## Changes in v1.219.029

- **Ocarina of Time closing as Kokiri Forest loads** (`tools/eglantine-patches/0082-…`): the log of issue #21 shows the session ending with `VK_ERROR_OUT_OF_DEVICE_MEMORY` while the pool still had a 322 MiB free block. The request that failed was a single buffer of 1,065,418,752 bytes, made while binding the storage buffers of one of the game's compute programs. The word after a storage buffer's pointer was taken as the buffer's size whenever it fitted inside the mapping; there it is most likely the float `1.0` (`0x3F800000`), not a size. A word above 256 MiB is now treated as not being a size, and the buffer is bound like a bare pointer (up to 8 MiB). When that happens the log says `Storage buffer size word … is not a size: bound as a bare pointer`.
- Not tested on a console yet. The Kokiri Forest load still needs a lot of memory right after this point, so a console that was already short may stop later for that reason.

## Changes in v1.219.028

- **Memory given back in one step** (`src/memory_pages.cpp`, `src/host_memory.cpp`): memory went back to the system by unmapping its range and then reserving the range again. In between, a mapping the system placed there (the zeroing view of the game's memory reserve) could be overwritten by the reservation, and the app crashed in `FillReserve`. Now a fixed reservation replaces the mapping in one call (`ReserveOver`). This build gives memory back rarely, so the crash was seen only in an experimental build that did it often.
- **Measurements and conclusions of the memory and CPU experiments** (builds 021-027, never released): [docs/MEMORY-CPU-EXPERIMENTS.md](docs/MEMORY-CPU-EXPERIMENTS.md). Their code is on branch `experiment/memoria-cpu`.

## Install

1. Download the release ZIP from **[Releases](../../releases)** and extract it. It contains one folder, `PPSA98515`.
2. Copy that folder to `/data/homebrew/PPSA98515` on the PS5 (or another folder ShadowMountPlus scans, e.g. `/mnt/usb0/homebrew/PPSA98515`). Use [ShadowMountPlus 1.7beta4](https://github.com/drakmor/ShadowMountPlus/releases/tag/1.7beta4) or newer.
3. Put your own **keys, firmware and games** in a *game files folder* (layout and options: [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md#install)). Games go in `roms/`: `.nsp`, `.xci`, **`.nup`**, or an extracted folder.
4. Start **EglantinePS5** from the PS5 home screen.

> [!IMPORTANT]
> This build uses the same title ID as the upstream app (`PPSA98515`), so it **replaces** it. From `v1.206.001` on, the in-app updater checks this repository: when a new release is out it asks first, then downloads it, checks its signature and hash, and replaces the app's files in the folder it is installed in. Earlier builds still ask prosperostore.net, so install `v1.206.001` by hand.

## Version numbers

`v1.206.001` is `1`, Eglantine's `build-206`, and this project's own count, which starts again at `001` with each new core build. The console's content version is the same thing as `01.206.001`.

## Using NUP files

- **Play:** drop a `.nup` in `roms/`. It shows in the Library like any game. A NUP holds the base game, its updates and DLC in one file.
- **Make one from a folder:** for a game kept as an extracted folder, open its settings in the Library (Triangle) and use the **NUP** row. Press twice to confirm; packing runs in the background and the `.nup` appears next to the folder when done. Updates and DLC are included.
- The format and the packer are part of the core (`src/core/file_sys/nup_container.*`, `nup_writer.*`).

## What this repository changes

All changes to the emulator core are plain patches in [`tools/eglantine-patches/`](tools/eglantine-patches), applied automatically at build time; if one stops applying, the build says so.

| Patch | Purpose |
|---|---|
| 0001 | Extracted dumps with `base/` + `updates/<version>/` folders |
| 0002 | Ampere U16 float-to-int conversion source modifiers (shader recompiler) |
| 0004 | NUP packer lists folders through `ListDirectory` (native directory reader on the console) |
| 0005 | Log why an NSO or NUP fails to load |
| 0008 | `nx2:vi2` command 201 stub (Switch 2) |
| 0009 | Guest-thread watchdog written to `eden_log.txt` (diagnosis of hangs) |
| 0010 | Do not load/save the global Vulkan static pipeline cache |
| 0011 | The DLC inside a NUP reaches the game through a content provider (`aoc:u`, data storage by ID) |
| 0012 | Donkey Kong Bananza: command lists that are empty when submitted are read when parsed |
| 0014 | Fence downloads: their staging buffers are reused as soon as the fence passes (Bananza ran out of graphics memory) |
| 0015 | Donkey Kong Bananza's upscaler slots (from [Orchid](https://github.com/MythosOfs/Orchid) `48d369310`, MythosOfs) |
| 0016 | Draws that sample their own colour target read a copy (Orchid `a49c81063`, MythosOfs) |
| 0017 | Page checks skipped for buffer ranges with no new CPU writes (Orchid `a3be6e11b`, `5ffe9eb53`, `4476c8671`, `2e0b577bb`; from monteger8/Eglantine-DKB, Claude for adamd) |
| 0018 | GPU split: wakes a back thread that stayed parked, also for the idle-hook request the core added in PR 639 (Orchid `1e4e0e82b`, from godpow/Eglantine, godpow) |
| 0019 | Log a guest thread that waits over 5 s for the GPU thread |
| 0020 | Log a kernel call that refuses an enum value (games abort on it with 2001-0120) |
| 0021 | Donkey Kong Bananza's terrain counters (heydemoura/EglantinePS5 patch `0008`) |
| 0022 | Direct guest reads of pages the GPU never wrote (from heydemoura/EglantinePS5 patch `0006`) |
| 0023 | Diagnostic headers the frontend's reports use (heydemoura/EglantinePS5 patch `0004`) |
| 0024 | Room in the GPU command queue for a burst of sparse tile maps, which froze Mario Kart World; each queue entry keeps fewer list entries in place, so the queue stays small |
| 0025 | Oversized native vertex streams are clamped (MythosOfs/Orchid `d7267d7f13`, godpow) |
| 0026 | Syncpoint increments with an id past the table are dropped (Orchid `bb9bdb1ed3`) |
| 0027 | Switch 2 scale, chat and camera commands are answered (Orchid `96dd790493`) |
| 0028 | Module names of Switch 2 programs (Orchid `0c92e5dd0e`) |
| 0029 | Scalar guest reads and writes split at 4 KiB device pages (Orchid `b841a347f2`) |
| 0030 | Syncpoint actions run outside the guard (Orchid `aa266e4577`, from andrewleeog/Eglantine `9ef2dde1a`) |
| 0032 | Tessellation pipelines the driver would refuse are skipped (Djihads80/Eglantine PR 426, kraibse) |
| 0033 | Texture cache counts what host images really take (Orchid `86587ae279`, godpow) |
| 0034 | Sparse images are backed whole when their tiles or layers cannot be set up (Orchid `64a5e4df86`) |
| 0035 | Released video images return as end-of-stream frames after a movie ends (Orchid `73cbde2a3a`) |
| 0036 | Native lowering of the HMMA matrix instruction, far less shader code (heydemoura/EglantinePS5 patch `0013`, adapted to this core) |
| 0037 | Controller callbacks are registered once the applet resource exists: a motion sample during boot crashed the app |
| 0038 | Task and mesh stages get the push constant range (Orchid `faa12fdbb2`, from Zortos and Eglantine PR 472) |
| 0039 | Reused GPU memory ranges are mapped again when their pages changed (Aubépine `92fe737e62`, from Purebuilt/AMDEGL `f6fba4f`) |
| 0040 | The syncpoint page acknowledges requests against values read after them (jpasqui `04f849c9bb`, without its trace) |
| 0041 | The GPU thread wakes for new submissions while channels wait on semaphores (Orchid `b15e4257b5`) |
| 0042 | Each memory manager's reference counts are sized to its own region (Orchid `d47bb55f34`, from exploitizm) |
| 0043 | Syncpoint actions are identified by id, so a late cancel cannot remove a newer one (Orchid `8837a265cc`, from Zortos and Eglantine PR 472; tests left out) |
| 0044 | The Legend of Zelda: Ocarina of Time's native programs and upscale stand-in (Eglantine PR 614, from A-Man-With-A-Plan-dev) |
| 0045 | Ocarina of Time's GPU-only regions are backed whole (Aubépine `24b7b61718`, adapted) |
| 0046 | Mario Kart World runs without task and mesh draws, which hang the PS5's GPU |
| 0047 | Mario Kart World's present pacing follows the presenting process (Eglantine PR 609; its buffer sizing part is in the core since PR 571) |
| 0048 | Command lists submitted behind a queued GPU map are copied after the map, not as zeros (Eglantine PR 543, merged with this core's empty-list handling) |
| 0049 | Fence releases from the puller run on the split GPU thread's back thread (Aubépine `7e20d36e9e`, after Lichii #70: Ocarina of Time stopped after its intro) |
| 0050 | Buffer flushes the front GPU thread needs run on the back thread, which owns the Vulkan scheduler (Aubépine `4d3b54834e`, after Lichii `12db3f599`; Donkey Kong Bananza crashed in `CopyBuffer`, Eglantine issue 636) |
| 0051 | Ocarina of Time's upscaling networks run through the native HMMA lowering instead of the plain upscale stand-in (its movies were black); `EDEN_NX2_TENSOR=0` turns this off |
| 0052 | Multiplies by 0 and 1 are folded so Ocarina of Time's sky ambient finds its buffer; pipeline cache versions bumped (Orchid `636c5fb385`) |
| 0053 | The GPU thread profile (`EDEN_NX2_GPU_PROFILE`) is read again when the GPU thread starts, so the testing switches can turn it on |
| 0055 | GPU memory layout ranges cached until the page table changes (Jonas, `47b4169b3f`) |
| 0056 | At fences only the Preflushable parts of a GPU write are copied back, the rest traps reads (Jonas, `5d742e369e`) |
| 0057 | Switch 2 games: CPU writes onto GPU-written pages go into the buffer instead of copying the page back first, which waited for the GPU; `EDEN_NX2_INLINE_PAGE_WRITES=0` turns it off (Zarigis, `575d0a5bfd`) |
| 0058 | The split GPU thread's front runs its sync fences, macro DMA read flushes and GPU memory maps on the back thread (kraibse with Duck912 and Zortos, `3087c1cf07`, merged with 0049 and 0050) |
| 0059 | Deferred work is submitted before waiting on our own fence (kraibse, `1b594aa6b6`) |
| 0060 | With delayed fence operations, fences are released again while an acquire waits (kraibse, `90ff431884`) |
| 0061 | GPU data is not copied back over newer CPU writes (Duck912, `fe2cd8be2b`) |
| 0062 | 4096 continuous range answers kept per thread (Duck912, `292a68df46`) |
| 0063 | Buffer usage masks for empty and partial blocks (zortos293, `5778922694`) |
| 0064 | Untouched buffer usage trackers are not reset (zortos293, `1b7995d89c`) |
| 0065 | Unchanged words are skipped while binding clean buffers (zortos293, `1725659895`) |
| 0066 | Storage buffer bindings cached per page table serial (Matt Artist, from Frizti/Lichii `ddf7952a14` via Duck912, `c8415d0e80`; adapted to this core's binding sizes) |
| 0067 | Only compute and DMA methods that write memory count as split writers (Frizti, `dc5938b8bb`) |
| 0068 | GPU-written page marks batched by bitmap word (zortos293, `abb9ae8426`) |
| 0069 | The last memory layout interval is reused during draw setup (zortos293, `d713f4462a`) |
| 0070 | With the GPU thread profile on, the guest thread watchdog writes which GPU thread stopped going round its loop and the zone it stopped in |
| 0071 | Task and mesh pipelines are kept in their own disk cache file with their mesh state, so they are not built again every session (Duck912, `bc121caf3e`, based on Frizti/Lichii `3a2c8980ad`) |
| 0072 | Layers of a large layered texture are uploaded in groups of up to 128 MB, each finished before the next reuses its staging buffer: four of Mario Kart World's 85 MB terrain layers asked for 352 MB at once and ran the race out of memory |
| 0073 | The GPU thread profile splits "push" into command lists, puller methods, channel binds and list ends, and a GPU thread that comes back with a new game takes its old watchdog place (the old one gave a false alarm) |
| 0074 | ReinterpretImage copies a combined depth-stencil image through its depth aspect only, and to the layout the image is in (cr1ckt, Lichii #89; it lost the device on NVIDIA) |
| 0075 | The alpha test reference is rounded to 1/255 in the pipeline key, so a fading reference no longer builds a new pipeline every frame (zortos293's port of Eglantine PR #544) |
| 0076 | The occlusion counter is paused, not reset, at each submission (Kaj, Lichii #121; Ocarina of Time's white flashes) |
| 0077 | Nested returns are counted before demotes move to the end of a fragment program (Inc0rporate, Lichii #127; black scenes on AMD GPUs) |
| 0078 | A new render target that a clear overwrites whole is not uploaded from guest memory first (Kaj, Lichii #141) |
| 0079 | Draws recorded after a graphics pipeline that failed to build are dropped instead of reaching the driver (Felipe Lima, Lichii #142) |
| 0080 | Dirty memory gatherers are serialized: video decoding also gathers off the GPU thread (Paulo Tumabing, core part of Lichii #138; a VP9 movie crash) |
| 0081 | No lost wake when worker tasks are cancelled (Zortos, core part of Lichii #290) |
| 0083 | Ocarina of Time runs without task and mesh draws, like Mario Kart World (0046): with them, every run stopped right after its intro movie |
| 0084 | ASTC arrays are re-compressed to BC1/BC3 at most 64 MiB of layers at a time (`EDEN_NX2_TRANSCODE_BATCH_MIB`): Ocarina of Time's 1365x1365, 90-layer array needed 894 MB plus 224 MB at once (core patch 0089 of branch `experiment/memoria-cpu`) |
| 0085 | That re-compression keeps its output buffer until its last batch (core patch 0090 of the same branch: build 01.219.026 crashed in `vkCmdCopyBufferToImage` without it) |

0074-0081 come from [zortos293/Lichii](https://github.com/zortos293/Lichii) (`nx2`), an Eglantine-based emulator. 0055-0069 and 0071 come from Eglantine's branch `oot-optimization-fixes` (Djihads80/Eglantine), not yet in `nx2`; its idle spin commit `8dc1e6e9c4` is left out, because the PS5 frontend already spins about 100 µs before a guest core sleeps, and its front flush commit `727a9a4eb4` is 0050. Number 0006 is intentionally missing: 32 storage buffers per stage is now in the core itself. Frontend changes (NUP in the library, "pack to NUP", `.gitmodules` pointing at Djihads80's core, Vulkan-port anchors for the newest core, per-line log flushing) are ordinary commits on top of the upstream frontend.

Logs on the console: `/data/eglantineps5/user/log/eden_log.txt` (and `.old.txt`), `/data/eglantineps5/logs/`. Settings: `/data/eglantineps5/config/eglantineps5.json`.

**Testing switches:** Settings > Diagnostics > Testing switches changes the emulator's switches (`EDEN_NX2_...`) from the menu: upscaling networks, task and mesh draws, the split GPU thread, the texture memory limit, the texture cache log, the GPU thread profile, CPU writes on GPU pages and others; each shows which of its choices is the default. They are saved to `/data/eglantineps5/config/env.txt` (one `NAME=VALUE` per line, which can also be edited by hand) and apply from the next start, so the app asks to confirm and closes.

## Updating and building

Update order, every time: **(1)** move the `Eglantine/` submodule to the newest Djihads80 `nx2` commit and re-apply the patches (`tools/update-eglantine.sh`; needs bash ≥ 4 because `apply-eglantine-patches.sh` uses `mapfile`, so on macOS use Homebrew bash or a Linux container); **(2)** merge the newest heydemoura/EglantinePS5 `main`; **(3)** check NUP loading and packing; **(4)** build, test on the console, publish.

Building follows [docs/BUILDING.md](docs/BUILDING.md) (`make package`, Linux x86-64 toolchain). On Apple Silicon we build in an `linux/amd64` Docker container (about 45 minutes for a clean build). Releases are the folder `PPSA98515/` zipped; the release notes list the core commit and the `eboot.bin` SHA-256.

## Credits and license

GPL-3.0-or-later (see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). The complete source of every release is this repository plus the pinned `Eglantine/` submodule commit.

Thanks to: **Djihads80 and the Eglantine contributors**; the **Eden** and yuzu projects; **heydemoura** (EglantinePS5 frontend and PS5 port work); **BlackBearReloaded** (ProsperoEden, ps5-opengl, PS5 native app boilerplate); **Mihawk** (PS5 Mesa/RADV and PS5 Vulkan); **mpereiraesaa** (ps5-vulkan); **drakmor** (ShadowMountPlus). The detailed credit list is in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

---

# Español

## Qué es

Una build homebrew para PS5 que apila tres capas, siempre actualizadas en este orden:

| # | Capa | Origen | Qué hacemos |
|---|---|---|---|
| 1 | **Núcleo del emulador** | [Djihads80/Eglantine](https://github.com/Djihads80/Eglantine), rama `nx2` (fork de Eden con soporte Switch 2) | Submódulo `Eglantine/` fijado al **último** commit de `nx2`. No lo forkeamos: nuestros cambios son parches pequeños. |
| 2 | **Frontend de PS5** | [heydemoura/EglantinePS5](https://github.com/heydemoura/EglantinePS5) (lanzador, port de Vulkan/RADV, empaquetado), basado en ProsperoEden de BlackBearReloaded | El historial de este repo es el de ese proyecto; rebaseamos sobre su `main` y mantenemos nuestros cambios encima. |
| 3 | **Soporte NUP** | `.nup` es el paquete de un solo fichero de Eglantine (juego base + update + DLC) | La biblioteca lista y ejecuta `.nup` y puede **empaquetar una carpeta de juego en un NUP** en la propia consola. |

Core actual: **Eglantine `0ca2032`** (`nx2`, `build-219` + 35 commits, 5-nov-2026).

A Switch 2 game (title IDs that start with `0400`) starts from the emulator core's own defaults for its games instead of the general values: **Faster GPU emulation** on and **Reactive flushing** off. Without them the core's second GPU thread stays off and every GPU fence waits for the console's GPU; Mario Kart World's title screen ran at about 11 FPS with the general defaults and about 24 FPS with these (see [docs/MKW-TITLE-FPS.md](docs/MKW-TITLE-FPS.md)). The game's own values, set in the Library, still win.

A game can have its own values: in its settings, **Performance** (see [Settings per game](#settings-per-game)). In the settings file, `"performance"` at the top level is what the switches set, and one inside a game's entry under `"games"` is that game's, whose values go first:

## Estado (versión honesta)

| Punto | Estado |
|---|---|
| Probado en consola | PS5 con firmware **13.60**, con jailbreak y ShadowMountPlus; otros usuarios la usan en **11.40** y **11.60**. |
| v1.219.029 | Compilada a partir de la v1.219.028 (mismo core) con un arreglo: Ocarina of Time ya no pide un búfer de 1 GiB al cargar el Bosque Kokiri, que era lo que cerraba el juego ahí (issue #21). **Compilada, aún sin probar en consola.** |
| v1.219.028 | Compilada el 8-11-2026 a partir de la v1.219.020 (mismo core) con un arreglo: la memoria que se devuelve al sistema queda reservada en un solo paso (una carrera que cerraba una build experimental). |
| Pruebas de memoria y CPU (builds 021-027, no publicadas) | Llevar la memoria del juego al SSD y mimalloc como heap de la app no ayudaron; dibujar los task y mesh shaders con compute sí. Mediciones y conclusiones (en inglés, con resumen en español): [docs/MEMORY-CPU-EXPERIMENTS.md](docs/MEMORY-CPU-EXPERIMENTS.md). |
| Donkey Kong Bananza (Switch 2, `.nup` con update 3.2.0 + DLC) | Funciona con la build actual (core `3081c8e`) con su update y su DLC: **modo TV, resolución 1x, 10-25 fps según la zona, unos 15 en zonas abiertas grandes, 30 fps en cinemáticas**. Una zona nueva puede verse morada la primera vez que se entra; en una prueba, la opción del juego «resetear terreno» lo arregló. |
| Modo portátil, resolución 0,5x | Sin confirmar. Los fallos anteriores venían de una caché de shaders antigua (ver abajo), así que hay que repetir la prueba. |
| Update y DLC dentro de un NUP | Se cargan los dos y la biblioteca los muestra. Comprobado con Bananza: el juego monta su DLC (log: `NUP add-on 040074A01BF13001 mounted`) y su menú de DLC muestra 3.2.0 y DLC 1.0.0. Los niveles del DLC aún no se han jugado a fondo. |
| NUP de Super Mario Bros. Wonder («Switch 2 Edition + DLC») | El archivo probado se ejecuta como el juego de Switch 1 con la update 1.2.0, y su DLC queda sin usar: los ejecutables de ese archivo son la versión de Switch 1 y el DLC contiene datos de Switch 2. La biblioteca lo sigue listando. |
| Mario Kart World (Switch 2) | Con **Emulación de GPU más rápida** activada para el juego, la v1.219.018 llega a la carrera a 25-43 fps (resolución 0,5x, modo portátil); sin ella, la partida va a unos 25 de sus 30 fps (v1.219.016). Los dibujos task y mesh están desactivados porque cuelgan la GPU de la PS5. Hasta ahora todas las partidas acaban a los uno a tres minutos: sin memoria (en carrera el juego ocupa unos 5,9 GB) o, una vez, colgado con el hilo de la GPU parado. El cierre en carrera fue una petición de 352 MB para cuatro capas de terreno de 85 MB; la v1.219.019 las sube en grupos de 128 MB como máximo, cosa que funcionó, y después el JIT no encontró un bloque de 16 MB para su caché de código con la reserva de 512 MB del juego sin usar. La v1.219.020 da al JIT su memoria en bloques de 2 MB, presta parte de esa reserva al JIT y a una petición gráfica rechazada (siempre queda la mitad para el juego) y devuelve los búferes de subida sin uso cuando falta memoria (sin probar aún). Reinicia la app antes de cada intento. |
| The Legend of Zelda: Ocarina of Time (Switch 2) | Se juega a unos 18 fps en la v1.219.015 (en la consola de un probador): se ven sus vídeos y el juego. Sus redes de escalado ocupan buena parte del tiempo de la GPU. |
| Donkey Kong Bananza en la v1.219.014 | Se cerraba al romperse muchos objetos a la vez; arreglado desde la v1.219.015. Una partida en la v1.219.016 fue a 12-15 fps sin ese cierre. |
| Otros juegos de Switch 2 | Algunos fallan pronto. No prometemos nada más allá de lo que aparece aquí. |
| Generación de fotogramas (Lossless Scaling) | Incluida, pero sin sentido con una base de 20 fps. |

### Problema conocido y arreglado: «se queda al 98% en cada arranque después del primero»

En la consola, el driver de Vulkan colgaba un juego de Switch 2 cuando arrancaba desde la caché de pipelines guardada por una ejecución anterior. Esta build ni carga ni guarda esa caché global (`tools/eglantine-patches/0010-…`); las cachés por juego siguen funcionando. Si alguna vez ves un cuelgue al 98%, borra `/data/eglantineps5/user/cache/shader/` y avísanos (adjunta `eden_log.txt`).

## Instalación

1. Descarga el ZIP desde **[Releases](../../releases)** y descomprímelo. Contiene una carpeta, `PPSA98515`.
2. Copia esa carpeta a `/data/homebrew/PPSA98515` en la PS5 (o a otra carpeta que escanee ShadowMountPlus, p. ej. `/mnt/usb0/homebrew/PPSA98515`). Usa [ShadowMountPlus 1.7beta4](https://github.com/drakmor/ShadowMountPlus/releases/tag/1.7beta4) o posterior.
3. Pon tus **claves, firmware y juegos** en una *carpeta de archivos de juego* (estructura y opciones: [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md#install), en inglés). Los juegos van en `roms/`: `.nsp`, `.xci`, **`.nup`** o una carpeta extraída.
4. Abre **EglantinePS5** desde la pantalla de inicio de la PS5.

> [!IMPORTANT]
> Esta build usa el mismo ID de título que la app original (`PPSA98515`), así que la **sustituye**. Desde `v1.206.001`, el actualizador integrado consulta este repositorio: cuando sale una release nueva pregunta primero, la descarga, comprueba su firma y su hash, y sustituye los archivos de la app en la carpeta donde esté instalada. Las builds anteriores siguen consultando prosperostore.net, así que instala `v1.206.001` a mano.

## Números de versión

`v1.206.001` es `1`, la `build-206` de Eglantine y un contador propio de este proyecto, que vuelve a `001` con cada build nueva del core. En la consola, la versión de contenido es lo mismo: `01.206.001`.

## Usar ficheros NUP

- **Jugar:** deja un `.nup` en `roms/`. Aparece en la biblioteca como cualquier juego. Un NUP contiene el juego base, sus updates y DLC en un fichero.
- **Crear uno desde una carpeta:** para un juego guardado como carpeta extraída, abre sus ajustes en la biblioteca (Triángulo) y usa la fila **NUP**. Pulsa dos veces para confirmar; el empaquetado se hace en segundo plano y el `.nup` aparece junto a la carpeta al terminar. Incluye updates y DLC.
- El formato y el empaquetador forman parte del core (`src/core/file_sys/nup_container.*`, `nup_writer.*`).

## Qué cambia este repositorio

Todos los cambios al core del emulador son parches simples en [`tools/eglantine-patches/`](tools/eglantine-patches), que se aplican solos al compilar; si uno deja de aplicar, la build lo dice. La tabla de parches está en la sección en inglés. Falta el 0006 a propósito: las 32 storage buffers por etapa ya vienen en el core. Los cambios del frontend (NUP en la biblioteca, «empaquetar a NUP», `.gitmodules` apuntando al core de Djihads80, anclajes del port de Vulkan para el core más nuevo, volcado del log línea a línea) son commits normales encima del frontend original.

Logs en la consola: `/data/eglantineps5/user/log/eden_log.txt` (y `.old.txt`), `/data/eglantineps5/logs/`. Ajustes: `/data/eglantineps5/config/eglantineps5.json`.

**Interruptores de prueba:** Ajustes > Diagnóstico > Interruptores de prueba cambia desde el menú los interruptores del emulador (`EDEN_NX2_...`): redes de escalado, dibujos task y mesh, el hilo de GPU dividido, el límite de memoria de texturas, el registro de texturas, el perfil del hilo de GPU, las escrituras de CPU en páginas de GPU y otros; cada uno indica cuál de sus opciones es la de por defecto. Se guardan en `/data/eglantineps5/config/env.txt` (un `NOMBRE=VALOR` por línea, que también se puede editar a mano) y se aplican desde el siguiente arranque, así que la app pide confirmación y se cierra.

## Actualizar y compilar

Orden de actualización, siempre: **(1)** subir el submódulo `Eglantine/` al último commit de `nx2` de Djihads80 y reaplicar los parches (`tools/update-eglantine.sh`; necesita bash ≥ 4 porque `apply-eglantine-patches.sh` usa `mapfile`, así que en macOS usa bash de Homebrew o un contenedor Linux); **(2)** fusionar el último `main` de heydemoura/EglantinePS5; **(3)** comprobar la carga y el empaquetado de NUP; **(4)** compilar, probar en consola y publicar.

La compilación sigue [docs/BUILDING.md](docs/BUILDING.md) (`make package`, toolchain Linux x86-64). En Apple Silicon compilamos dentro de un contenedor Docker `linux/amd64` (unos 45 minutos una build limpia). Las releases son la carpeta `PPSA98515/` comprimida; las notas indican el commit del core y el SHA-256 del `eboot.bin`.

## Créditos y licencia

GPL-3.0-or-later (ver [LICENSE](LICENSE) y [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). El código fuente completo de cada release es este repositorio más el commit fijado del submódulo `Eglantine/`.

Gracias a: **Djihads80 y los colaboradores de Eglantine**; los proyectos **Eden** y yuzu; **heydemoura** (frontend EglantinePS5 y port a PS5); **BlackBearReloaded** (ProsperoEden, ps5-opengl, boilerplate nativo de PS5); **Mihawk** (PS5 Mesa/RADV y PS5 Vulkan); **mpereiraesaa** (ps5-vulkan); **drakmor** (ShadowMountPlus). La lista de créditos detallada está en [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).
