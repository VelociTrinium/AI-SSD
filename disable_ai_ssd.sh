#!/bin/bash
# AI-SSD V2 Firmware & Emulation Device Teardown
# STRICT REQUIREMENTS:
# - Shell script ONLY disables/stops the AI-SSD firmware/device.
# - MUST NOT run Python, inference, or benchmarks.
# - Gracefully stops the AI-SSD/QEMU management process.
# - Cleans up project-owned runtime state in /tmp/ai-ssd-runtime.
# - Safe to run when firmware is already disabled (idempotent).
# - Does NOT terminate unrelated QEMU processes.
# - Does NOT alter real kernel modules or host NVMe devices.
# - Verifies device/process stopped and outputs standardized status banner.

RUN_DIR="/tmp/ai-ssd-runtime"
PID_FILE="${RUN_DIR}/qemu.pid"
PORT=9999
STOP_TIMEOUT_S=10

echo "========================================"
echo "        AI-SSD V2 FIRMWARE"
echo "========================================"

# Check if guest daemon port is currently open
is_port_open() {
    (exec 3<>/dev/tcp/127.0.0.1/"$PORT") >/dev/null 2>&1
    local ret=$?
    exec 3>&- 2>/dev/null || true
    return $ret
}

# 1. Identify if AI-SSD process is running via PID file
TARGET_PID=""
if [ -f "$PID_FILE" ]; then
    TARGET_PID=$(cat "$PID_FILE" 2>/dev/null | tr -d '[:space:]')
fi

# If PID file is missing or invalid, check if port is open
if [ -z "$TARGET_PID" ] || ! kill -0 "$TARGET_PID" 2>/dev/null; then
    if ! is_port_open; then
        # Already completely stopped
        rm -rf "$RUN_DIR"
        echo "AI-SSD Firmware: DISABLED"
        echo "Status:          ALREADY STOPPED"
        echo "Firmware is disabled."
        echo "========================================"
        exit 0
    fi

    # Port is open but PID file was missing - identify specific AI-SSD QEMU instance safely
    # Only match QEMU command line containing serial=v2-ai-ssd-001 or hostfwd=tcp:127.0.0.1:9999
    TARGET_PID=$(pgrep -f "serial=v2-ai-ssd-001" 2>/dev/null | head -n 1)
    if [ -z "$TARGET_PID" ]; then
        TARGET_PID=$(pgrep -f "127.0.0.1:${PORT}-:9999" 2>/dev/null | head -n 1)
    fi
fi

if [ -z "$TARGET_PID" ] || ! kill -0 "$TARGET_PID" 2>/dev/null; then
    rm -rf "$RUN_DIR"
    echo "AI-SSD Firmware: DISABLED"
    echo "Status:          ALREADY STOPPED"
    echo "Firmware is disabled."
    echo "========================================"
    exit 0
fi

# Double check the PID belongs to an AI-SSD process before sending signal
CMDLINE=$(tr '\0' ' ' < /proc/"$TARGET_PID"/cmdline 2>/dev/null || echo "")
if [[ "$CMDLINE" != *"v2-ai-ssd-001"* && "$CMDLINE" != *"${PORT}-:9999"* ]]; then
    echo "AI-SSD Firmware: DISABLE FAILED"
    echo "Reason: Process $TARGET_PID does not match AI-SSD instance signature"
    echo "========================================"
    exit 1
fi

# 2. Gracefully terminate process (SIGTERM)
kill -15 "$TARGET_PID" 2>/dev/null || true

# Wait for process to exit and port to close
ELAPSED=0
STOPPED=0
while [ $ELAPSED -lt $STOP_TIMEOUT_S ]; do
    if ! kill -0 "$TARGET_PID" 2>/dev/null; then
        STOPPED=1
        break
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

# If still alive after SIGTERM timeout, send SIGKILL
if [ $STOPPED -eq 0 ]; then
    kill -9 "$TARGET_PID" 2>/dev/null || true
    sleep 1
    if ! kill -0 "$TARGET_PID" 2>/dev/null; then
        STOPPED=1
    fi
fi

# 3. Clean up runtime directory and verify port is free
rm -rf "$RUN_DIR"

if is_port_open; then
    echo "AI-SSD Firmware: DISABLE FAILED"
    echo "Reason: Port $PORT remains active after process termination"
    echo "========================================"
    exit 1
fi

if [ $STOPPED -eq 1 ]; then
    echo "AI-SSD Firmware: DISABLED"
    echo "NVMe Device:     /dev/nvme0n1"
    echo "Status:          STOPPED"
    echo "Firmware is disabled."
    echo "========================================"
    exit 0
else
    echo "AI-SSD Firmware: DISABLE FAILED"
    echo "Reason: Unable to terminate process $TARGET_PID"
    echo "========================================"
    exit 1
fi
