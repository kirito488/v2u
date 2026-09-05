"""
下载并整理 val 数据（requests 分块下载，更鲁棒）

策略:
  每个文件最多尝试 5 次:
    requests 分块下载（断点续传） → zstd 校验 → 解压 → 整理
  失败自动重试
"""
import os
import re
import sys
import shutil
import subprocess
import tarfile
import zstandard
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_URL = "https://huggingface.co/datasets/VJiaLi/V2U4Real/resolve/main/val"
VAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2u4real", "val")
CHUNK_SIZE = 8 * 1024 * 1024  # 8MB

# val 所有文件（场景__agent）
VAL_FILES = [
    "2025-07-17-16-12_7__agent-1", "2025-07-17-16-12_7__agent-2",
    "2025-07-17-16-12_8__agent-1", "2025-07-17-16-12_8__agent-2",
    "2025-07-17-16-35_2__agent-1", "2025-07-17-16-35_2__agent-2",
    "2025-07-17-16-50_6__agent-1", "2025-07-17-16-50_6__agent-2",
    "2025-07-17-17-07_1__agent-1", "2025-07-17-17-07_1__agent-2",
    "2025-07-17-17-42_7__agent-1", "2025-07-17-17-42_7__agent-2",
    "2025-07-18-12-10_4__agent-1", "2025-07-18-12-10_4__agent-2",
    "2025-07-18-12-37_5__agent-1", "2025-07-18-12-37_5__agent-2",
    "2025-07-18-12-53_4__agent-1", "2025-07-18-12-53_4__agent-2",
]


def agent_complete(scene, agent):
    agent_dir = os.path.join(VAL_DIR, scene, agent)
    yaml_dir = os.path.join(agent_dir, "yaml", "ouster")
    lidar_dir = os.path.join(agent_dir, "ouster")
    yaml_cnt = len(os.listdir(yaml_dir)) if os.path.isdir(yaml_dir) else 0
    lidar_cnt = len(os.listdir(lidar_dir)) if os.path.isdir(lidar_dir) else 0
    return yaml_cnt > 0 and lidar_cnt > 0


def download_requests(url, local):
    """requests 分块下载，支持断点续传"""
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    resume = os.path.getsize(local) if os.path.exists(local) else 0
    headers = {'Range': f'bytes={resume}-'} if resume > 0 else {}

    try:
        resp = session.get(url, stream=True, timeout=(30, 300), headers=headers)
        if resp.status_code == 416:
            return True  # 已完整
        if resp.status_code not in (200, 206):
            print(f"    HTTP {resp.status_code}")
            return False

        mode = 'ab' if resume > 0 else 'wb'
        with open(local, mode) as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"    下载异常: {type(e).__name__}: {e}")
        return False


def verify_zstd(local):
    r = subprocess.run(f'zstd -t "{local}" 2>&1', shell=True,
                       capture_output=True, text=True, timeout=300)
    return r.returncode == 0


def extract(local, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    try:
        with open(local, 'rb') as fh:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                with tarfile.open(fileobj=reader, mode='r|') as tar:
                    tar.extractall(path=extract_dir)
        return True
    except Exception as e:
        print(f"    解压失败: {type(e).__name__}: {e}")
        return False


def organize(extract_dir, scene, agent):
    inner_agent = None
    for root, dirs, files in os.walk(extract_dir):
        if os.path.basename(root) == agent:
            has_lidar = any(d in ['ouster', 'ruby', 'm1'] for d in dirs)
            has_yaml = 'yaml' in dirs
            if has_lidar or has_yaml:
                inner_agent = root
                break
    if inner_agent is None:
        return False

    target = os.path.join(VAL_DIR, scene, agent)
    os.makedirs(target, exist_ok=True)
    moved = 0
    for entry in os.listdir(inner_agent):
        src = os.path.join(inner_agent, entry)
        dst = os.path.join(target, entry)
        if os.path.exists(dst):
            continue
        shutil.move(src, dst)
        moved += 1
    return moved > 0


def main():
    os.makedirs(VAL_DIR, exist_ok=True)
    todo = []
    for fname in VAL_FILES:
        m = re.match(r'^(.*)__agent-(\d+)$', fname)
        if not m:
            continue
        scene, agent = m.group(1), m.group(2)
        if agent_complete(scene, agent):
            print(f"[跳过] {scene}/agent-{agent} 已完整")
            continue
        todo.append((scene, agent, fname))

    print(f"\n需要下载: {len(todo)} 个文件\n")
    done = 0
    failed = []

    for scene, agent, fname in todo:
        local_tar = os.path.join(VAL_DIR, f"{fname}.tar.zst")
        extract_dir = os.path.join(VAL_DIR, f"_tmp_{fname}")
        url = f"{BASE_URL}/{fname}.tar.zst"

        success = False
        for attempt in range(1, 6):  # 最多 5 次
            if attempt > 1:
                print(f"    (重试 {attempt}/5)")
            if os.path.isdir(extract_dir):
                shutil.rmtree(extract_dir, ignore_errors=True)
            # 保留已下载的 local_tar 用于续传，校验失败才删除

            print(f"[{done+1}/{len(todo)}] {fname}.tar.zst 下载中..."
                  f"({os.path.getsize(local_tar)/1024**2:.0f}MB 已有)" if os.path.exists(local_tar) else
                  f"[{done+1}/{len(todo)}] {fname}.tar.zst 下载中...")
            if not download_requests(url, local_tar):
                continue

            size = os.path.getsize(local_tar) / 1024**3
            if not verify_zstd(local_tar):
                print(f"    FAIL 校验失败 ({size:.2f}GB), 删除重下")
                os.remove(local_tar)
                continue

            if not extract(local_tar, extract_dir):
                print("    FAIL 解压失败")
                continue

            if not organize(extract_dir, scene, agent):
                print("    FAIL 整理失败")
                continue

            shutil.rmtree(extract_dir, ignore_errors=True)
            os.remove(local_tar)
            done += 1
            print(f"    OK {scene}/agent-{agent} 完成")
            success = True
            break

        if not success:
            failed.append(fname)
            print(f"    FAIL {fname} 5次均失败")

    print("\n" + "=" * 50)
    print(f"完成: {done}/{len(todo)}")
    if failed:
        print(f"失败: {len(failed)}")
        for f in failed:
            print(f"  - {f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
