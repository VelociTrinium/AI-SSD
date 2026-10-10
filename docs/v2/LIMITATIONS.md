# AI-SSD V2: Limitations & Non-Claims

**Scope**: Explicit technical boundary documentation, virtualization artifacts, hardware assumptions, model constraints, scaling bottlenecks, and negative results.

---

## 1. Virtualization & Device Environment Constraints

### 1.1 QEMU Virtual NVMe Device Overhead
All virtual device benchmarks were conducted using a virtual NVMe controller emulated by QEMU 6.2.0 (`qemu-system-x86_64`) running with KVM hardware acceleration on an AWS EC2 instance.
- **Trap-and-Emulate Overhead**: Every NVMe I/O request traverses user-space QEMU device emulation, virtio socket networking, and the host Linux kernel.
- **Virtual vs Bare-Metal Latency**: In our benchmarks, the virtual NVMe stack incurs a **$2.33\times$ latency penalty** compared to direct file I/O on the host filesystem ($20.92\text{ s}$ vs $8.98\text{ s}$ for 16 decode steps at 4,096 context).
- **Socket IPC Bottleneck**: Commands sent from the host Python process to the in-guest C daemon (`nvme_guest_daemon.c`) pass over a localhost TCP socket (`127.0.0.1:9999`). While `TCP_NODELAY` is enabled, this transport incurs minor IPC serialization overhead that would not exist on a native PCIe memory-mapped BAR register.

---

## 2. Hardware Non-Claims

To ensure absolute scientific and commercial integrity, the authors explicitly state what was **NOT** measured:

### 2.1 No Physical Computational SSD Silicon
- We did **not** manufacture or test custom ASIC silicon, FPGA prototypes, or commercially packaged computational storage drives (such as Samsung SmartSSD, ScaleFlux CSD, or Solidigm computational prototypes).
- All controller-side vector dot products were executed by the host CPU operating inside the QEMU KVM guest virtual machine.

### 2.2 No PCIe Electrical Measurements
- We did **not** capture physical electrical bus traces using an oscilloscope, PCIe protocol analyzer, or hardware logic analyzer.
- Data movement metrics (such as the 81.5% PCIe traffic reduction and 0-byte candidate Key invariant) represent **exact byte counts of data transferred across the host-storage software interface boundary**, not physical electrical waveforms.

### 2.3 No Energy / Power Measurements
- We did **not** measure physical wattage, joules, or voltage rails on physical server hardware.
- Claims such as "reduces system energy by X%" or "cuts storage power by Y%" are **unsupported by empirical data** and must not be asserted.

### 2.4 No Physical NAND Flash Wear / Cell Degradation Data
- We did **not** measure physical NAND program/erase (P/E) cycles, oxide breakdown, or raw bit error rates (RBER).
- Multi-channel FTL load balancing metrics (0.86% load imbalance) were derived from deterministic logical coordinate mapping, not silicon endurance stress testing.

---

## 3. Computational & Execution Constraints

### 3.1 CPU-Only Inference
- All model weights (`Qwen/Qwen3-4B-Instruct-2507` and `Qwen/Qwen3-8B`) were evaluated on **Intel Xeon Platinum 8488C host CPU cores**.
- On CPU, compute-heavy operators (such as PyTorch `aten::linear` inside MLP layers) account for **60.4% ($12.64\text{ s}$)** of total decode wall time.
- In GPU accelerated environments (e.g., NVIDIA H100/A100) where matrix operations execute in microseconds, storage retrieval latency would represent a substantially larger fraction of the critical path unless paired with ultra-high-throughput enterprise flash channels.

### 3.2 Thread-Scaling Wall
- Performance peaks strictly at 4 threads on an 8-vCPU node.
- Because the physical node possesses 4 physical execution cores (with 2 hyperthreads per core), scaling beyond 4 threads (6 and 8 threads) introduces thread synchronization contention and cache thrashing without adding hardware execution units, causing throughput to plateau or degrade.

---

## 4. Extended Context Scaling (32K Latency Bottleneck)

While AI-SSD successfully bounds physical memory at 32,768 context (**saving 27.90 GB of RAM** and maintaining flat process RSS):
- **Sequential Software Scan Bottleneck**: At 32,768 context, there are 2,048 candidate blocks per layer, totaling **73,728 blocks across all 36 layers**.
- In the virtual guest daemon, scanning and computing dot products across 73,728 blocks sequentially using 2 virtual CPU cores takes substantial wall-clock time.
- As a result, decode throughput drops from **0.765 tok/s at 4K context** down to **0.024 tok/s at 32K context** ($679.13\text{ s}$ for 16 decode tokens).
- **Architectural Takeaway**: To maintain interactive throughput ($> 1\text{ tok/s}$) at 32K context and beyond, in-storage Top-$K$ filtering cannot rely on general-purpose software CPU cores; it requires **massively parallel hardware ASIC vector matrix engines** or dedicated near-flash DSP accelerators.

---

## 5. Model Architecture & KV Layout Compatibility

1. **Fixed Block Size Geometry**: The KV manager relies on fixed 16-token blocks aligned to physical 4 KiB/8 KiB page multiples. Models with unusual head dimensions that do not align cleanly to 64-byte or 4,096-byte boundaries require zero-padding, causing slight storage fragmentation.
2. **GQA Assumption**: The scoring kernels and FTL channel distribution algorithms assume modern Grouped-Query Attention ($H_Q \ge H_{KV}$). Standard Multi-Head Attention (where $H_Q = H_{KV}$) multiplies candidate Key storage volume by the GQA ratio ($4\times$ to $7\times$).
3. **Sparse Attention Quality**: While 10% Top-$K$ selection produced 100% exact token matches for the tested 16-token generation sequences, extreme long-form text generation ($> 1,000$ generated tokens) under strict Top-$K$ pruning may experience attention drift or semantic degradation unless paired with dynamic eviction strategies or adaptive thresholding.
4. **Sliding-Window and Token Horizon**: Models with sliding-window attention (such as Mistral) constrain attention to a localized window of tokens. Tokens falling outside the sliding window cannot influence attention and are culled from active candidate selection; storing them in deep flash is valuable only if the model implements global retrieval mechanisms.
5. **Non-Separable State Models (Pure SSM / RNN)**: State-space models (e.g. Mamba, RWKV) maintain a compact $O(1)$ hidden state that is updated recurrence-wise rather than indexing a growing history of Key-Value vectors. Because there is no separable Key-Value cache to prune or retrieve via Top-$K$ selection, pure SSM models cannot leverage AI-SSD computational storage and are classified as `UNSUPPORTED`. Hybrid models (such as Jamba) are classified as `PARTIAL` because only attention layers benefit from offload.
