"""
V2U4Real 数据集下载脚本
从 Hugging Face (镜像) 下载 V2U4Real 数据集
"""
import os
import sys
import time
from huggingface_hub import snapshot_download

# 使用国内镜像加速下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

REPO_ID = 'VJiaLi/V2U4Real'
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v2u4real')

def download_dataset():
    print(f"开始下载 V2U4Real 数据集...")
    print(f"仓库: {REPO_ID}")
    print(f"目标目录: {LOCAL_DIR}")
    print(f"使用镜像: {os.environ.get('HF_ENDPOINT', 'N/A')}")
    print("-" * 60)

    os.makedirs(LOCAL_DIR, exist_ok=True)

    start_time = time.time()

    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type='dataset',
            local_dir=LOCAL_DIR,
            resume_download=True,       # 支持断点续传
            max_workers=4,              # 并行下载数
        )
        elapsed = time.time() - start_time
        print(f"\n✓ 下载完成! 耗时: {elapsed/60:.1f} 分钟")
        print(f"数据保存在: {LOCAL_DIR}")

        # 打印下载的文件结构
        print("\n文件结构:")
        for root, dirs, files in os.walk(LOCAL_DIR):
            level = root.replace(LOCAL_DIR, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            if level > 2:  # 限制深度
                continue
            sub_indent = ' ' * 2 * (level + 1)
            for file in sorted(files)[:5]:
                filepath = os.path.join(root, file)
                size_mb = os.path.getsize(filepath) / (1024**2)
                print(f'{sub_indent}{file} ({size_mb:.1f} MB)')
            if len(files) > 5:
                print(f'{sub_indent}... 还有 {len(files) - 5} 个文件')

    except KeyboardInterrupt:
        print("\n下载被中断。重新运行脚本可以继续下载（断点续传）。")
        sys.exit(1)
    except Exception as e:
        print(f"\n下载出错: {e}")
        print("重新运行脚本可以继续下载（断点续传）。")
        sys.exit(1)


def download_by_split(split_name: str):
    """按 train/val/test 分别下载"""
    print(f"下载 {split_name} 部分...")
    print("-" * 40)

    local_split_dir = os.path.join(LOCAL_DIR, split_name)
    os.makedirs(local_split_dir, exist_ok=True)

    start_time = time.time()

    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type='dataset',
            local_dir=LOCAL_DIR,
            allow_patterns=[f"{split_name}/*"],
            resume_download=True,
            max_workers=4,
        )
        elapsed = time.time() - start_time
        print(f"✓ {split_name} 下载完成! 耗时: {elapsed/60:.1f} 分钟")
    except Exception as e:
        print(f"× {split_name} 下载出错: {e}")
        raise


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # 按部分下载
        split = sys.argv[1]  # train / val / test
        download_by_split(split)
    else:
        # 下载全部
        download_dataset()
