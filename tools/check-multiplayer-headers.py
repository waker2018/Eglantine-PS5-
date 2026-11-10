#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Check active Ninja objects agree on the derived multiplayer class declarations."""
import pathlib
import subprocess
import sys

build = pathlib.Path(sys.argv[1])
def ninja(*args):
    return subprocess.run(['ninja', '-C', str(build), '-t', *args],
                          capture_output=True, text=True, check=True).stdout

active = {line.split(': ', 1)[0] for line in ninja('targets', 'all').splitlines()}
headers = ('network/packet.h', 'network/room_member.h',
           'core/internal_network/sockets.h', 'core/internal_network/socket_proxy.h',
           'core/hle/service/sockets/bsd.h')
# A core that hardens its room sockets itself (AcceptsProxyPacket in socket_proxy.h) keeps its own
# socket_proxy.h and bsd.h: headless/multiplayer_packets.cmake derives them only for older cores.
# Every object then reads the original, which is as consistent as every object reading ours.
core_owned = ('core/internal_network/socket_proxy.h', 'core/hle/service/sockets/bsd.h')
seen = dict.fromkeys(headers, 0)
original = {header: [] for header in headers}
wrong, target = [], None
for line in ninja('deps').splitlines():
    if line and not line.startswith(' '):
        target = line.split(': ', 1)[0]
        continue
    if target not in active:
        continue
    path = line.strip().replace('\\', '/')
    for header in headers:
        if path.endswith('/' + header):
            prefix = '/headless/multiplayer/' + ('socket-headers/' if header.startswith('core/') else '')
            if path.endswith(prefix + header):
                seen[header] += 1
            else:
                original[header].append(f'{target}: {path}')
core_kept = [header for header in core_owned if not seen[header] and original[header]]
for header in headers:
    if header not in core_kept:
        wrong += original[header]
missing = [header for header, count in seen.items() if not count and header not in core_kept]
if core_kept:
    print(f'Multiplayer headers kept from the core: {core_kept}')
if wrong or missing:
    print('\n'.join(wrong[:20]))
    sys.exit(f'Multiplayer headers: {len(wrong)} original-header dependencies, missing {missing}')
print(f'Multiplayer derived headers: {seen} PASS')
