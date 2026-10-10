"""Automated build script for the native in-storage attention C kernel.

Supports Linux x86_64 GCC (with AVX2/FMA) and Windows 64-bit MSVC / MinGW.
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path


def compile_c_kernel() -> Path:
    kernel_dir = Path(__file__).parent.resolve()
    c_source = kernel_dir / "instorage_attention.c"

    if not c_source.exists():
        raise FileNotFoundError(f"Source file {c_source} not found.")

    is_windows = platform.system() == "Windows"
    out_name = "instorage_attention.dll" if is_windows else "instorage_attention.so"
    out_path = kernel_dir / out_name

    if is_windows:
        # Check for MSVC BuildTools vcvarsall.bat
        vcvars_paths = [
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
            r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
        ]

        for vcvars in vcvars_paths:
            if os.path.exists(vcvars):
                cmd = f'cmd.exe /c "call "{vcvars}" x64 && cl /O2 /LD /GS- "{c_source}" /Fe:"{out_path}" /link /NOENTRY /NODEFAULTLIB"'
                print(f"Compiling with MSVC x64: {cmd}")
                res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res.returncode == 0 and out_path.exists():
                    print(f"[SUCCESS] Native C kernel compiled to {out_path}")
                    return out_path
                else:
                    print(f"[WARNING] MSVC build failed: {res.stderr}")

    # GCC build (Linux or MinGW on Windows)
    gcc_path = shutil.which("gcc") or (r"C:\MinGW\bin\gcc.exe" if os.path.exists(r"C:\MinGW\bin\gcc.exe") else None)
    if gcc_path:
        flags = [gcc_path, "-O3", "-shared", "-fPIC", "-std=c99", "-Wall"]
        if platform.machine() in ("x86_64", "AMD64"):
            flags.extend(["-mavx2", "-mfma"])
        flags.extend([str(c_source), "-o", str(out_path)])
        print(f"Compiling with GCC: {' '.join(flags)}")
        res = subprocess.run(flags, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0 and out_path.exists():
            print(f"[SUCCESS] Native C kernel compiled to {out_path}")
            return out_path
        else:
            print(f"[WARNING] GCC build failed: {res.stderr}")

    print("[INFO] Using existing precompiled library or Python fallback.")
    return out_path


if __name__ == "__main__":
    compile_c_kernel()
