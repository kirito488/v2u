"""
Download and extract V2U4Real train split (requests + resume + zstd + organize).

Usage:
  python download_train.py
  HF_ENDPOINT=https://hf-mirror.com python download_train.py   # mirror (default below)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tarfile

import requests
import zstandard
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Prefer HF mirror on CN servers; override with env HF_ENDPOINT if needed.
HF = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
BASE_URL = f"{HF}/datasets/VJiaLi/V2U4Real/resolve/main/train"
TRAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2u4real", "train")
CHUNK_SIZE = 8 * 1024 * 1024

TRAIN_FILES = [
    "2025-07-17-16-12_1__agent-1", "2025-07-17-16-12_1__agent-2",
    "2025-07-17-16-12_2__agent-1", "2025-07-17-16-12_2__agent-2",
    "2025-07-17-16-12_3__agent-1", "2025-07-17-16-12_3__agent-2",
    "2025-07-17-16-12_4__agent-1", "2025-07-17-16-12_4__agent-2",
    "2025-07-17-16-12_5__agent-1", "2025-07-17-16-12_5__agent-2",
    "2025-07-17-16-35_3__agent-1", "2025-07-17-16-35_3__agent-2",
    "2025-07-17-16-50_1__agent-1", "2025-07-17-16-50_1__agent-2",
    "2025-07-17-16-50_2__agent-1", "2025-07-17-16-50_2__agent-2",
    "2025-07-17-16-50_3__agent-1", "2025-07-17-16-50_3__agent-2",
    "2025-07-17-17-07_2__agent-1", "2025-07-17-17-07_2__agent-2",
    "2025-07-17-17-07_6__agent-1", "2025-07-17-17-07_6__agent-2",
    "2025-07-17-17-42_1__agent-1", "2025-07-17-17-42_1__agent-2",
    "2025-07-17-17-42_2__agent-1", "2025-07-17-17-42_2__agent-2",
    "2025-07-17-17-42_3__agent-1", "2025-07-17-17-42_3__agent-2",
    "2025-07-17-17-42_4__agent-1", "2025-07-17-17-42_4__agent-2",
    "2025-07-17-17-42_5__agent-1", "2025-07-17-17-42_5__agent-2",
    "2025-07-18-12-10_1__agent-1", "2025-07-18-12-10_1__agent-2",
    "2025-07-18-12-10_2__agent-1", "2025-07-18-12-10_2__agent-2",
    "2025-07-18-12-10_3__agent-1", "2025-07-18-12-10_3__agent-2",
    "2025-07-18-12-37_1__agent-1", "2025-07-18-12-37_1__agent-2",
    "2025-07-18-12-37_2__agent-1", "2025-07-18-12-37_2__agent-2",
    "2025-07-18-12-37_3__agent-1", "2025-07-18-12-37_3__agent-2",
    "2025-07-18-12-53_1__agent-1", "2025-07-18-12-53_1__agent-2",
    "2025-07-18-12-53_2__agent-1", "2025-07-18-12-53_2__agent-2",
    "2025-07-18-12-53_3__agent-1", "2025-07-18-12-53_3__agent-2",
]


def agent_complete(scene: str, agent: str) -> bool:
    agent_dir = os.path.join(TRAIN_DIR, scene, agent)
    yaml_dir = os.path.join(agent_dir, "yaml", "ouster")
    lidar_dir = os.path.join(agent_dir, "ouster")
    yaml_cnt = len(os.listdir(yaml_dir)) if os.path.isdir(yaml_dir) else 0
    lidar_cnt = len(os.listdir(lidar_dir)) if os.path.isdir(lidar_dir) else 0
    return yaml_cnt > 0 and lidar_cnt > 0


def download_requests(url: str, local: str) -> bool:
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    resume = os.path.getsize(local) if os.path.exists(local) else 0
    headers = {"Range": f"bytes={resume}-"} if resume > 0 else {}
    try:
        resp = session.get(url, stream=True, timeout=(30, 300), headers=headers)
        if resp.status_code == 416:
            return True
        if resp.status_code not in (200, 206):
            print(f"    HTTP {resp.status_code}")
            return False
        mode = "ab" if resume > 0 else "wb"
        with open(local, mode) as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"    download error: {type(e).__name__}: {e}")
        return False


def verify_zstd(local: str) -> bool:
    r = subprocess.run(
        f'zstd -t "{local}" 2>&1', shell=True,
        capture_output=True, text=True, timeout=600,
    )
    return r.returncode == 0


def extract(local: str, extract_dir: str) -> bool:
    os.makedirs(extract_dir, exist_ok=True)
    try:
        with open(local, "rb") as fh:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    tar.extractall(path=extract_dir)
        return True
    except Exception as e:
        print(f"    extract error: {type(e).__name__}: {e}")
        return False


def organize(extract_dir: str, scene: str, agent: str) -> bool:
    inner_agent = None
    for root, dirs, _files in os.walk(extract_dir):
        if os.path.basename(root) == agent:
            has_lidar = any(d in ["ouster", "ruby", "m1"] for d in dirs)
            has_yaml = "yaml" in dirs
            if has_lidar or has_yaml:
                inner_agent = root
                break
    if inner_agent is None:
        return False
    target = os.path.join(TRAIN_DIR, scene, agent)
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
    os.makedirs(TRAIN_DIR, exist_ok=True)
    todo = []
    for fname in TRAIN_FILES:
        m = re.match(r"^(.*)__agent-(\d+)$", fname)
        if not m:
            continue
        scene, agent = m.group(1), m.group(2)
        if agent_complete(scene, agent):
            print(f"[skip] {scene}/agent-{agent} complete")
            continue
        todo.append((scene, agent, fname))

    print(f"\nTo download: {len(todo)} archives -> {TRAIN_DIR}\n")
    done = 0
    failed = []

    for scene, agent, fname in todo:
        local_tar = os.path.join(TRAIN_DIR, f"{fname}.tar.zst")
        extract_dir = os.path.join(TRAIN_DIR, f"_tmp_{fname}")
        url = f"{BASE_URL}/{fname}.tar.zst"
        success = False
        for attempt in range(1, 6):
            if attempt > 1:
                print(f"    retry {attempt}/5")
            if os.path.isdir(extract_dir):
                shutil.rmtree(extract_dir, ignore_errors=True)
            if os.path.exists(local_tar):
                print(f"[{done+1}/{len(todo)}] {fname}.tar.zst "
                      f"({os.path.getsize(local_tar)/1024**2:.0f}MB partial)")
            else:
                print(f"[{done+1}/{len(todo)}] {fname}.tar.zst")
            if not download_requests(url, local_tar):
                continue
            if not verify_zstd(local_tar):
                print("    FAIL zstd check, remove and retry")
                os.remove(local_tar)
                continue
            if not extract(local_tar, extract_dir):
                continue
            if not organize(extract_dir, scene, agent):
                print("    FAIL organize")
                continue
            shutil.rmtree(extract_dir, ignore_errors=True)
            os.remove(local_tar)
            done += 1
            print(f"    OK {scene}/agent-{agent}")
            success = True
            break
        if not success:
            failed.append(fname)
            print(f"    FAIL {fname} after 5 attempts")

    print("\n" + "=" * 50)
    print(f"done: {done}/{len(todo)}")
    if failed:
        print(f"failed: {len(failed)}")
        for f in failed:
            print(f"  - {f}")
    print("=" * 50)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
