"""
整理 V2U4Real 数据为 OpenCOOD 格式

原始结构:
  v2u4real/train/场景__agent-1/train/场景/1/{ouster,ruby,m1,yaml}
  v2u4real/train/场景__agent-2/train/场景/2/{ouster,yaml}
  v2u4real/test/场景__agent-2.tar.zst

目标结构:
  v2u4real/train/场景/1/{ouster,ruby,m1,yaml}
  v2u4real/train/场景/2/{ouster,yaml}
  v2u4real/test/场景/1/...
  v2u4real/test/场景/2/...
"""
import os
import re
import shutil
import tarfile
import zstandard

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v2u4real')


def organize_split(split):
    """整理 train/val: 合并 agent 文件夹"""
    split_dir = os.path.join(ROOT, split)
    if not os.path.isdir(split_dir):
        print(f"[跳过] {split_dir} 不存在")
        return

    # 找所有 "场景__agent-N" 文件夹
    agent_dirs = sorted(os.listdir(split_dir))
    print(f"\n[{split}] 找到 {len(agent_dirs)} 个条目")

    for item in agent_dirs:
        item_path = os.path.join(split_dir, item)
        if not os.path.isdir(item_path):
            continue

        # 解析场景名和 agent id
        m = re.match(r'^(.*)__agent-(\d+)$', item)
        if not m:
            print(f"  [跳过] 不是 agent 格式: {item}")
            continue

        scen_name, agent_id = m.group(1), m.group(2)
        target_scen_dir = os.path.join(split_dir, scen_name)
        target_agent_dir = os.path.join(target_scen_dir, agent_id)

        # 找到实际的 agent 数据目录: item/train/场景/agent_id/
        # 可能有多层嵌套
        inner_agent = None
        for root, dirs, files in os.walk(item_path):
            if os.path.basename(root) == agent_id:
                # 确认这是 agent 数据目录（含 ouster 或 yaml）
                has_lidar = any(d in ['ouster', 'ruby', 'm1'] for d in dirs)
                has_yaml = 'yaml' in dirs
                if has_lidar or has_yaml:
                    inner_agent = root
                    break

        if inner_agent is None:
            print(f"  [警告] {item} 找不到 agent-{agent_id} 数据目录")
            continue

        # 移动到目标位置
        os.makedirs(target_agent_dir, exist_ok=True)
        moved = 0
        for entry in os.listdir(inner_agent):
            src = os.path.join(inner_agent, entry)
            dst = os.path.join(target_agent_dir, entry)
            if os.path.exists(dst):
                print(f"  [跳过] {dst} 已存在")
                continue
            shutil.move(src, dst)
            moved += 1

        print(f"  [OK] {scen_name}/agent-{agent_id}: 移动 {moved} 项")

        # 删除原始 agent 文件夹
        shutil.rmtree(item_path, ignore_errors=True)

    # 清理空的 __agent 残留
    print(f"  [{split}] 整理完成")


def extract_test():
    """解压 test 的 tar.zst"""
    test_dir = os.path.join(ROOT, 'test')
    if not os.path.isdir(test_dir):
        print(f"[跳过] {test_dir} 不存在")
        return

    print(f"\n[test] 检查压缩文件...")
    for f in os.listdir(test_dir):
        if f.endswith('.tar.zst'):
            fpath = os.path.join(test_dir, f)
            m = re.match(r'^(.*)__agent-(\d+)\.tar\.zst$', f)
            if not m:
                continue
            scen_name, agent_id = m.group(1), m.group(2)
            print(f"  解压: {f}")

            dctx = zstandard.ZstdDecompressor()
            out_dir = os.path.join(test_dir, f"{scen_name}__agent-{agent_id}")
            os.makedirs(out_dir, exist_ok=True)

            with open(fpath, 'rb') as fh:
                with dctx.stream_reader(fh) as reader:
                    with tarfile.open(fileobj=reader, mode='r|') as tar:
                        tar.extractall(path=out_dir)

            print(f"  解压完成 → {out_dir}")

            # 删除压缩文件
            os.remove(fpath)
            print(f"  已删除压缩文件")


if __name__ == '__main__':
    organize_split('train')
    organize_split('val')
    extract_test()
    print("\n=== 全部完成 ===")
