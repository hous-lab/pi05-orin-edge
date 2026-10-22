#!/bin/bash
# Start or restart all 4 transport stubs on orin (CPU-only, no GPU).
# Usage: bash start_orin_stubs.sh   (run via ssh from x86; returns immediately)
D=${TRANSPORT_BENCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
B=${PYTHON_BIN:-python3}
cd $D
for pid in $(ps aux | grep echo_server | grep -v grep | awk '{print $2}'); do kill $pid 2>/dev/null; done
sleep 0.5
setsid $B echo_server.py --transport ws --port 8311 >echo_ws.log 2>&1 </dev/null &
setsid $B echo_server.py --transport tcp --port 8312 >echo_tcp.log 2>&1 </dev/null &
setsid $B echo_server.py --transport uds >echo_uds.log 2>&1 </dev/null &
setsid $B echo_server.py --transport shm >echo_shm.log 2>&1 </dev/null &
sleep 1
for f in echo_ws.log echo_tcp.log echo_uds.log echo_shm.log; do tail -n1 $f; done
exit 0
