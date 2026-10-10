#!/bin/bash
# AI-SSD V2 Firmware & Emulation Device Initializer
# STRICT REQUIREMENTS:
# - Shell script ONLY enables/initializes the AI-SSD firmware/device.
# - MUST NOT run Python, inference, or benchmarks.
# - Verifies device availability and outputs standardized status banner.

RUN_DIR="/tmp/ai-ssd-runtime"
PID_FILE="${RUN_DIR}/qemu.pid"
RAW_IMG="/opt/ai-ssd-v2/images/v2_nvme.raw"
INITRD="/opt/ai-ssd-v2/images/initramfs.cpio.gz"
KERNEL="/opt/ai-ssd-v2/images/vmlinuz"
PORT=9999
TIMEOUT_S=30

echo "========================================"
echo "        AI-SSD V2 FIRMWARE"
echo "========================================"

mkdir -p "$RUN_DIR"

# 1. Prerequisite Checks
if [ ! -f "$RAW_IMG" ]; then
    echo "AI-SSD Firmware: DISABLED"
    echo "Reason: Missing storage image $RAW_IMG"
    echo "========================================"
    exit 1
fi

if [ ! -f "$INITRD" ]; then
    echo "AI-SSD Firmware: DISABLED"
    echo "Reason: Missing initramfs $INITRD"
    echo "========================================"
    exit 1
fi

if [ ! -f "$KERNEL" ]; then
    echo "AI-SSD Firmware: DISABLED"
    echo "Reason: Missing kernel image $KERNEL"
    echo "========================================"
    exit 1
fi

if ! command -v qemu-system-x86_64 >/dev/null 2>&1; then
    echo "AI-SSD Firmware: DISABLED"
    echo "Reason: qemu-system-x86_64 not found in PATH"
    echo "========================================"
    exit 1
fi

# 2. Check if QEMU AI-SSD guest daemon is already running
if (exec 3<>/dev/tcp/127.0.0.1/"$PORT") >/dev/null 2>&1; then
    exec 3>&- 2>/dev/null || true
    echo "AI-SSD Firmware: ENABLED"
    echo "NVMe Device:     /dev/nvme0n1"
    echo "Status:          READY"
    echo ""
    echo "Firmware remains enabled."
    echo "Run the inference script manually."
    echo "========================================"
    exit 0
fi

LOG_FILE="${RUN_DIR}/qemu.log"

# Detect KVM availability (POSIX sh compatible)
if [ -e /dev/kvm ] && [ -w /dev/kvm ]; then
    KVM_FLAG="-enable-kvm"
    CPU_FLAG="host"
else
    # Fallback to software emulation if hardware KVM is not available (common on cloud VMs)
    KVM_FLAG=""
    CPU_FLAG="max"
    TIMEOUT_S=60
fi

# 3. Launch QEMU with NVMe Controller and Guest Daemon
nohup qemu-system-x86_64 \
    $KVM_FLAG \
    -cpu "$CPU_FLAG" \
    -m 2048 \
    -smp 2 \
    -no-reboot \
    -kernel "$KERNEL" \
    -initrd "$INITRD" \
    -drive file="$RAW_IMG",format=raw,if=none,id=nvme0 \
    -device nvme,drive=nvme0,serial=v2-ai-ssd-001,num_queues=8,logical_block_size=4096,physical_block_size=4096 \
    -netdev user,id=net0,hostfwd=tcp:127.0.0.1:${PORT}-:9999 \
    -device virtio-net-pci,netdev=net0 \
    -pidfile "$PID_FILE" \
    -nographic \
    -append "console=ttyS0 panic=-1 quiet loglevel=3 daemon=1" >"$LOG_FILE" 2>&1 &

# 4. Wait until the expected device is available via guest daemon port
ELAPSED=0
CONNECTED=0
while [ $ELAPSED -lt $TIMEOUT_S ]; do
    if (exec 3<>/dev/tcp/127.0.0.1/"$PORT") >/dev/null 2>&1; then
        exec 3>&- 2>/dev/null || true
        CONNECTED=1
        break
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

if [ $CONNECTED -eq 1 ]; then
    echo "AI-SSD Firmware: ENABLED"
    echo "NVMe Device:     /dev/nvme0n1"
    echo "Status:          READY"
    echo ""
    echo "Firmware remains enabled."
    echo "Run the inference script manually."
    echo "========================================"
    exit 0
else
    # Clean up process if launch failed / timed out
    if [ -f "$PID_FILE" ]; then
        Q_PID=$(cat "$PID_FILE" 2>/dev/null)
        if [ -n "$Q_PID" ] && kill -0 "$Q_PID" 2>/dev/null; then
            kill -15 "$Q_PID" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
    fi
    echo "AI-SSD Firmware: DISABLED"
    echo "Reason: Timeout waiting for NVMe guest daemon on port $PORT after ${TIMEOUT_S}s"
    if [ -f "$LOG_FILE" ] && [ -s "$LOG_FILE" ]; then
        echo "----------------------------------------"
        echo "QEMU Console Log:"
        tail -n 15 "$LOG_FILE"
        echo "----------------------------------------"
    fi
    echo "========================================"
    exit 1
fi
