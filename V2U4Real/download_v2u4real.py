"""
V2U4Real 数据集并行下载脚本
使用 hf-mirror resolve 端点，支持断点续传和多文件并行下载
"""
import os
import sys
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 配置
MIRROR_BASE = "https://hf-mirror.com/datasets/VJiaLi/V2U4Real/resolve/main"
LOCAL_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2u4real")
MAX_WORKERS = 4  # 并行下载数
CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks
TIMEOUT = 7200  # 单文件超时 (2 hours)

# 所有文件列表
FILES = [
    # train (38 files)
    "train/2025-07-17-16-12_1__agent-1.tar.zst",
    "train/2025-07-17-16-12_1__agent-2.tar.zst",
    "train/2025-07-17-16-12_2__agent-1.tar.zst",
    "train/2025-07-17-16-12_2__agent-2.tar.zst",
    "train/2025-07-17-16-12_3__agent-1.tar.zst",
    "train/2025-07-17-16-12_3__agent-2.tar.zst",
    "train/2025-07-17-16-12_4__agent-1.tar.zst",
    "train/2025-07-17-16-12_4__agent-2.tar.zst",
    "train/2025-07-17-16-12_5__agent-1.tar.zst",
    "train/2025-07-17-16-12_5__agent-2.tar.zst",
    "train/2025-07-17-16-35_3__agent-1.tar.zst",
    "train/2025-07-17-16-35_3__agent-2.tar.zst",
    "train/2025-07-17-16-50_1__agent-1.tar.zst",
    "train/2025-07-17-16-50_1__agent-2.tar.zst",
    "train/2025-07-17-16-50_2__agent-1.tar.zst",
    "train/2025-07-17-16-50_2__agent-2.tar.zst",
    "train/2025-07-17-16-50_3__agent-1.tar.zst",
    "train/2025-07-17-16-50_3__agent-2.tar.zst",
    "train/2025-07-17-17-07_2__agent-1.tar.zst",
    "train/2025-07-17-17-07_2__agent-2.tar.zst",
    "train/2025-07-17-17-07_6__agent-1.tar.zst",
    "train/2025-07-17-17-07_6__agent-2.tar.zst",
    "train/2025-07-17-17-42_1__agent-1.tar.zst",
    "train/2025-07-17-17-42_1__agent-2.tar.zst",
    "train/2025-07-17-17-42_2__agent-1.tar.zst",
    "train/2025-07-17-17-42_2__agent-2.tar.zst",
    "train/2025-07-17-17-42_3__agent-1.tar.zst",
    "train/2025-07-17-17-42_3__agent-2.tar.zst",
    "train/2025-07-17-17-42_4__agent-1.tar.zst",
    "train/2025-07-17-17-42_4__agent-2.tar.zst",
    "train/2025-07-17-17-42_5__agent-1.tar.zst",
    "train/2025-07-17-17-42_5__agent-2.tar.zst",
    "train/2025-07-18-12-10_1__agent-1.tar.zst",
    "train/2025-07-18-12-10_1__agent-2.tar.zst",
    "train/2025-07-18-12-10_2__agent-1.tar.zst",
    "train/2025-07-18-12-10_2__agent-2.tar.zst",
    "train/2025-07-18-12-10_3__agent-1.tar.zst",
    "train/2025-07-18-12-10_3__agent-2.tar.zst",
    "train/2025-07-18-12-37_1__agent-1.tar.zst",
    "train/2025-07-18-12-37_1__agent-2.tar.zst",
    "train/2025-07-18-12-37_2__agent-1.tar.zst",
    "train/2025-07-18-12-37_2__agent-2.tar.zst",
    "train/2025-07-18-12-37_3__agent-1.tar.zst",
    "train/2025-07-18-12-37_3__agent-2.tar.zst",
    "train/2025-07-18-12-53_1__agent-1.tar.zst",
    "train/2025-07-18-12-53_1__agent-2.tar.zst",
    "train/2025-07-18-12-53_2__agent-1.tar.zst",
    "train/2025-07-18-12-53_2__agent-2.tar.zst",
    "train/2025-07-18-12-53_3__agent-1.tar.zst",
    "train/2025-07-18-12-53_3__agent-2.tar.zst",
    # val (16 files)
    "val/2025-07-17-16-12_7__agent-1.tar.zst",
    "val/2025-07-17-16-12_7__agent-2.tar.zst",
    "val/2025-07-17-16-12_8__agent-1.tar.zst",
    "val/2025-07-17-16-12_8__agent-2.tar.zst",
    "val/2025-07-17-16-35_2__agent-1.tar.zst",
    "val/2025-07-17-16-35_2__agent-2.tar.zst",
    "val/2025-07-17-16-50_6__agent-1.tar.zst",
    "val/2025-07-17-16-50_6__agent-2.tar.zst",
    "val/2025-07-17-17-07_1__agent-1.tar.zst",
    "val/2025-07-17-17-07_1__agent-2.tar.zst",
    "val/2025-07-17-17-42_7__agent-1.tar.zst",
    "val/2025-07-17-17-42_7__agent-2.tar.zst",
    "val/2025-07-18-12-10_4__agent-1.tar.zst",
    "val/2025-07-18-12-10_4__agent-2.tar.zst",
    "val/2025-07-18-12-37_5__agent-1.tar.zst",
    "val/2025-07-18-12-37_5__agent-2.tar.zst",
    "val/2025-07-18-12-53_4__agent-1.tar.zst",
    "val/2025-07-18-12-53_4__agent-2.tar.zst",
    # test (12 files)
    "test/2025-07-17-16-12_6__agent-1.tar.zst",
    "test/2025-07-17-16-12_6__agent-2.tar.zst",
    "test/2025-07-17-16-35_1__agent-1.tar.zst",
    "test/2025-07-17-16-35_1__agent-2.tar.zst",
    "test/2025-07-17-16-50_4__agent-1.tar.zst",
    "test/2025-07-17-16-50_4__agent-2.tar.zst",
    "test/2025-07-17-16-50_5__agent-1.tar.zst",
    "test/2025-07-17-16-50_5__agent-2.tar.zst",
    "test/2025-07-17-17-07_3__agent-1.tar.zst",
    "test/2025-07-17-17-07_3__agent-2.tar.zst",
    "test/2025-07-17-17-42_6__agent-1.tar.zst",
    "test/2025-07-17-17-42_6__agent-2.tar.zst",
    "test/2025-07-17-17-42_8__agent-1.tar.zst",
    "test/2025-07-17-17-42_8__agent-2.tar.zst",
    "test/2025-07-18-12-10_5__agent-1.tar.zst",
    "test/2025-07-18-12-10_5__agent-2.tar.zst",
    "test/2025-07-18-12-37_4__agent-1.tar.zst",
    "test/2025-07-18-12-37_4__agent-2.tar.zst",
    "test/2025-07-18-12-53_5__agent-1.tar.zst",
    "test/2025-07-18-12-53_5__agent-2.tar.zst",
]

# 进度统计
stats_lock = threading.Lock()
stats = {"completed": 0, "failed": 0, "skipped": 0, "total_bytes": 0}

def format_size(size_bytes):
    if size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.2f} GB"

def format_speed(bytes_per_sec):
    return format_size(bytes_per_sec) + "/s"

def download_file(filepath, index, total):
    """Download a single file with resume support"""
    url = f"{MIRROR_BASE}/{filepath}"
    local_path = os.path.join(LOCAL_BASE, filepath)
    local_dir = os.path.dirname(local_path)

    # Create directory
    os.makedirs(local_dir, exist_ok=True)

    # Create session with retry
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    label = f"[{index}/{total}]"

    try:
        # Check existing file
        resume_pos = 0
        if os.path.exists(local_path):
            resume_pos = os.path.getsize(local_path)
            if resume_pos > 0:
                print(f"  {label} {filepath}: 已存在 {format_size(resume_pos)}, 续传中...")

        # Stream download with resume
        headers = {}
        if resume_pos > 0:
            headers['Range'] = f'bytes={resume_pos}-'

        response = session.get(url, stream=True, timeout=30, headers=headers)

        if response.status_code == 416:
            # Range not satisfiable - file already complete
            print(f"  ✓ {label} {filepath}: 已完成 (跳过)")
            with stats_lock:
                stats["skipped"] += 1
            return True

        if response.status_code not in (200, 206):
            print(f"  ✗ {label} {filepath}: HTTP {response.status_code}")
            with stats_lock:
                stats["failed"] += 1
            return False

        # Get total size
        if 'Content-Range' in response.headers:
            total_size = int(response.headers['Content-Range'].split('/')[-1])
            mode = 'ab'
        else:
            total_size = int(response.headers.get('Content-Length', 0))
            mode = 'wb'
            resume_pos = 0

        start_time = time.time()
        last_print_time = start_time
        bytes_since_print = 0

        with open(local_path, mode) as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    bytes_since_print += len(chunk)
                    resume_pos += len(chunk)

                    # Print progress every 30 seconds
                    now = time.time()
                    if now - last_print_time >= 30 and total_size > 0:
                        elapsed = now - start_time
                        speed = resume_pos / elapsed if elapsed > 0 else 0
                        pct = resume_pos * 100 / total_size if total_size else 0

                        # Estimate remaining time
                        if speed > 0:
                            remaining_bytes = total_size - resume_pos
                            eta = remaining_bytes / speed
                            eta_str = f"剩余 {eta/60:.0f}分"
                        else:
                            eta_str = ""

                        print(f"  {label} {filepath}: {pct:.0f}% "
                              f"({format_size(resume_pos)}/{format_size(total_size)}) "
                              f"{format_speed(speed)} {eta_str}")
                        last_print_time = now
                        bytes_since_print = 0

        # Verify local size
        local_size = os.path.getsize(local_path)
        if total_size > 0 and local_size >= total_size:
            elapsed = time.time() - start_time
            print(f"  ✓ {label} {filepath}: 完成 ({format_size(local_size)}, "
                  f"{format_speed(local_size/elapsed if elapsed > 0 else 0)})")
            with stats_lock:
                stats["completed"] += 1
                stats["total_bytes"] += local_size
            return True
        elif total_size > 0:
            print(f"  ⚠ {label} {filepath}: 大小不匹配 ({format_size(local_size)} vs {format_size(total_size)})")
            with stats_lock:
                stats["failed"] += 1
            return False
        else:
            # Unknown size, assume success
            elapsed = time.time() - start_time
            print(f"  ✓ {label} {filepath}: 完成 ({format_size(local_size)})")
            with stats_lock:
                stats["completed"] += 1
                stats["total_bytes"] += local_size
            return True

    except requests.exceptions.Timeout:
        print(f"  ⚠ {label} {filepath}: 超时 (已下载 {format_size(os.path.getsize(local_path) if os.path.exists(local_path) else 0)}，将重试)")
        with stats_lock:
            stats["failed"] += 1
        return False
    except Exception as e:
        print(f"  ✗ {label} {filepath}: {type(e).__name__}: {e}")
        with stats_lock:
            stats["failed"] += 1
        return False


def main():
    total = len(FILES)
    print("=" * 60)
    print(f"V2U4Real 数据集下载 (并行)")
    print(f"源: {MIRROR_BASE}")
    print(f"目标: {LOCAL_BASE}")
    print(f"文件数: {total}")
    print(f"并行数: {MAX_WORKERS}")
    print("=" * 60)

    overall_start = time.time()

    # Download in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_file, filepath, i+1, total): filepath
            for i, filepath in enumerate(FILES)
        }

        # Also periodically print overall progress
        last_overall_print = time.time()

        for future in as_completed(futures):
            filepath = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  ✗ 未处理异常 {filepath}: {e}")

            # Print overall progress
            now = time.time()
            if now - last_overall_print >= 60:  # Every minute
                with stats_lock:
                    done = stats["completed"] + stats["skipped"] + stats["failed"]
                    elapsed = now - overall_start
                    print(f"\n  ### 总体进度: {done}/{total} | "
                          f"成功: {stats['completed']} | "
                          f"跳过: {stats['skipped']} | "
                          f"失败: {stats['failed']} | "
                          f"总下载: {format_size(stats['total_bytes'])} | "
                          f"已用时: {elapsed/60:.0f}分 ###\n")
                last_overall_print = now

    # Final summary
    elapsed = time.time() - overall_start
    with stats_lock:
        print("\n" + "=" * 60)
        print(f"下载完成!")
        print(f"总文件: {total}")
        print(f"成功: {stats['completed']}")
        print(f"跳过 (已完成): {stats['skipped']}")
        print(f"失败: {stats['failed']}")
        print(f"总下载量: {format_size(stats['total_bytes'])}")
        print(f"总耗时: {elapsed/60:.0f} 分钟 ({elapsed/3600:.1f} 小时)")
        print("=" * 60)

        if stats['failed'] > 0:
            print("\n重新运行此脚本以重试失败的文件。")
            return 1
        return 0


if __name__ == '__main__':
    # 清空之前的 git LFS 指针文件
    if os.path.exists(LOCAL_BASE):
        import shutil
        print("清理旧的 git LFS 占位文件...")
        # 只删除 .tar.zst 文件 (LFS 指针), 保留目录结构
        for root, dirs, files in os.walk(LOCAL_BASE):
            for f in files:
                if f.endswith('.tar.zst'):
                    filepath = os.path.join(root, f)
                    if os.path.getsize(filepath) < 10000:  # LFS pointer is ~130 bytes
                        os.remove(filepath)
        print("清理完成。")

    sys.exit(main())
