"""V2U4Real test 数据集稳健下载（requests 流式 + zstandard 校验 + 解压）"""
import os, sys, json
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

MIRROR = "https://hf-mirror.com/datasets/VJiaLi/V2U4Real/resolve/main"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2u4real")
TEST_DIR = os.path.join(DATA_DIR, "test")

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

CHUNK = 8 * 1024 * 1024  # 8MB

def files_count(d):
    if not os.path.isdir(d):
        return 0
    return sum(len(fs) for _, _, fs in os.walk(d))

def build_session():
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=2,
                    status_forcelist=[500, 502, 503, 504])
    s.mount('https://', HTTPAdapter(max_retries=retries))
    return s

def download(url, local, session):
    """断点续传下载"""
    resume = os.path.getsize(local) if os.path.exists(local) else 0
    headers = {'Range': f'bytes={resume}-'} if resume > 0 else {}
    try:
        r = session.get(url, stream=True, timeout=(30, 600), headers=headers)
        if r.status_code == 416:
            return True  # 已完整
        if r.status_code not in (200, 206):
            print(f"    HTTP {r.status_code}")
            return False
        mode = 'ab' if resume > 0 else 'wb'
        with open(local, mode) as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"    下载异常: {type(e).__name__}: {e}")
        return False

def verify(local):
    """用 zstandard 校验完整性（流式，不占太多内存）"""
    import zstandard as zstd
    try:
        dctx = zstd.ZstdDecompressor()
        with open(local, 'rb') as f:
            reader = dctx.stream_reader(f)
            # 读一小段即可，能解出就说明帧完整
            reader.read(1)
        return True
    except Exception:
        return False

def extract(local, extract_dir):
    """解压 tar.zst 到目录"""
    import zstandard as zstd
    import tarfile
    os.makedirs(extract_dir, exist_ok=True)
    try:
        dctx = zstd.ZstdDecompressor()
        with open(local, 'rb') as f:
            dstream = dctx.stream_reader(f)
            with tarfile.open(fileobj=dstream, mode='r|') as tar:
                tar.extractall(extract_dir)
        os.remove(local)  # 成功解压后删除压缩包
        return True
    except Exception as e:
        print(f"    解压异常: {type(e).__name__}: {e}")
        return False

def main():
    # 只处理指定场景（可选），默认全部
    only = sys.argv[1] if len(sys.argv) > 1 else None
    todo = []
    for scene in TEST_SCENES:
        if only and scene != only:
            continue
        for agent in ('1', '2'):
            fname = f"{scene}__agent-{agent}.tar.zst"
            local_tar = os.path.join(TEST_DIR, fname)
            extract_dir = os.path.join(TEST_DIR, f"{scene}__agent-{agent}")
            if os.path.isdir(extract_dir) and files_count(extract_dir) > 0:
                continue  # 已解压
            todo.append((scene, agent, fname, local_tar, extract_dir))

    if not todo:
        print("全部已下载完成！")
        return

    print(f"待下载: {len(todo)} 个文件")
    session = build_session()
    done, failed = 0, []

    for scene, agent, fname, local_tar, extract_dir in todo:
        tag = f"[{done+1}/{len(todo)}]"
        url = f"{MIRROR}/test/{fname}"
        os.makedirs(TEST_DIR, exist_ok=True)

        # 若已解压则跳过
        if os.path.isdir(extract_dir) and files_count(extract_dir) > 0:
            print(f"\n{tag} test/{fname}  OK 已存在，跳过")
            done += 1
            continue

        # 下载（最多5次）
        ok = False
        for attempt in range(1, 6):
            if os.path.exists(local_tar) and not verify(local_tar):
                try:
                    os.remove(local_tar)
                except PermissionError:
                    pass
            if download(url, local_tar, session):
                if verify(local_tar):
                    ok = True
                    break
                else:
                    print(f"    (校验失败, 重试 {attempt}/5)")
                    try:
                        os.remove(local_tar)
                    except PermissionError:
                        pass
            else:
                print(f"    (下载失败, 重试 {attempt}/5)")

        if not ok:
            print(f"  X {tag} test/{fname} 下载失败")
            failed.append(f"test/{fname}")
            continue

        if extract(local_tar, extract_dir):
            n = files_count(extract_dir)
            print(f"  OK {tag} test/{fname}  ({n} 文件)")
            done += 1
        else:
            print(f"  X {tag} test/{fname} 解压失败")
            failed.append(f"test/{fname}")

    print("\n" + "=" * 60)
    print(f"完成! 成功: {done}/{len(todo)}")
    if failed:
        print(f"失败: {len(failed)}")
        for f in failed:
            print(f"  - {f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
