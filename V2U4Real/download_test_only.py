"""V2U4Real test 数据集下载+解压（只处理 test split）"""
import os, sys, subprocess, re, shutil
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

MIRROR = "https://hf-mirror.com/datasets/VJiaLi/V2U4Real/resolve/main"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2u4real")

# 只处理 test
TEST_SCENES = [
    "2025-07-17-16-12_6",
    "2025-07-17-16-35_1",
    "2025-07-17-16-50_4", "2025-07-17-16-50_5",
    "2025-07-17-17-07_3",
    "2025-07-17-17-42_6", "2025-07-17-17-42_8",
    "2025-07-18-12-10_5",
    "2025-07-18-12-37_4",
    "2025-07-18-12-53_5",
]

def build_todo():
    """构建待下载列表"""
    todo = []
    for scene in TEST_SCENES:
        for agent in ('1', '2'):
            fname = f"{scene}__agent-{agent}.tar.zst"
            hf_path = f"test/{fname}"
            local_tar = os.path.join(DATA_DIR, "test", fname)
            extract_dir = os.path.join(DATA_DIR, "test", f"{scene}__agent-{agent}")

            # 检查是否已解压
            if os.path.isdir(extract_dir):
                cnt = sum(1 for _, _, fs in os.walk(extract_dir) for _ in fs)
                if cnt > 0:
                    continue
            todo.append((scene, agent, fname, hf_path, local_tar, extract_dir))
    return todo

def files_count(d):
    if not os.path.isdir(d):
        return 0
    return sum(len(fs) for _, _, fs in os.walk(d))

def download(url, local):
    """下载单个文件，支持续传"""
    resume = os.path.getsize(local) if os.path.exists(local) else 0
    cmd = ["curl", "-s", "-L", "-C", "-", "--connect-timeout", "30",
           "--max-time", "7200", "-o", local, url]
    if resume > 0:
        print(f"    (续传 {resume/1024**2:.0f}MB)")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7500)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False

def verify(local):
    """zstd 完整性校验"""
    r = subprocess.run(
        f'zstd -t "{local}" 2>&1',
        shell=True, capture_output=True, text=True, timeout=300
    )
    return r.returncode == 0

def extract(local, extract_dir):
    """解压 tar.zst"""
    os.makedirs(extract_dir, exist_ok=True)
    r = subprocess.run(
        f'zstd -d -c "{local}" 2>/dev/null | tar xf - -C "{extract_dir}" 2>&1',
        shell=True, capture_output=True, text=True, timeout=600
    )
    if r.returncode == 0:
        os.remove(local)
        return True
    return False

def main():
    todo = build_todo()
    total = len(todo)
    if total == 0:
        print("test 全部已下载完成！")
        return
    print(f"test 待下载: {total} 个文件")

    done = 0
    failed = []

    for scene, agent, fname, hf_path, local_tar, extract_dir in todo:
        tag = f"[{done+1}/{total}]"
        url = f"{MIRROR}/{hf_path}"
        os.makedirs(os.path.dirname(local_tar), exist_ok=True)

        for attempt in range(1, 6):  # 最多5次重试
            if attempt > 1:
                print(f"    (重试 {attempt}/5)")
                if os.path.exists(local_tar):
                    os.remove(local_tar)

            if os.path.isdir(extract_dir) and files_count(extract_dir) > 0:
                print(f"\n{tag} test/{fname}  OK 已存在，跳过")
                done += 1
                break

            if not download(url, local_tar):
                print(f"    download 失败")
                continue

            if not verify(local_tar):
                size = os.path.getsize(local_tar) / 1024**2
                print(f"    X 校验失败 ({size:.0f}MB), 重新下载")
                os.remove(local_tar)
                continue

            if extract(local_tar, extract_dir):
                n = files_count(extract_dir)
                print(f"  OK [{done+1}/{total}] test/{fname}  ({n} 文件)")
                done += 1
                break
            else:
                print(f"    X 解压失败")
                os.remove(local_tar)
                continue
        else:
            print(f"  X [{done+1}/{total}] test/{fname} 5次均失败, 跳过")
            failed.append(f"test/{fname}")

    print("\n" + "=" * 60)
    print(f"完成! 成功: {done}/{total}")
    if failed:
        print(f"失败: {len(failed)}")
        for f in failed:
            print(f"  - {f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
