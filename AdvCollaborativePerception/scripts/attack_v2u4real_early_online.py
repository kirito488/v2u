"""早融合 spoof 入口：只改 UAV 点云，再用 attfuse/where2comm/coalign 提 BEV 并中间融合。

等价于:
    python scripts/attack_v2u4real_online.py --mode spoof --level early --model attfuse
"""
import os
import sys

root = os.path.normpath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../"))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "scripts"))

from attack_v2u4real_online import main as online_main


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--mode", "spoof", "--level", "early"] + sys.argv[1:]
    online_main()
