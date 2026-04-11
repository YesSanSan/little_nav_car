#!/usr/bin/env python3

import os
from pathlib import Path

def find_nested_directories_detailed(base_dir):
    """
    查找同名嵌套目录（详细版本）
    """
    base_path = Path(base_dir)
    nested_dirs = []

    if not base_path.exists():
        print(f"错误: 目录 {base_dir} 不存在")
        return nested_dirs

    if not base_path.is_dir():
        print(f"错误: {base_dir} 不是目录")
        return nested_dirs

    print(f"扫描目录: {base_dir}")

    for item in base_path.iterdir():
        if item.is_dir():
            nested_path = item / item.name
            if nested_path.exists() and nested_path.is_dir():
                print(f"找到嵌套目录: {item} -> {nested_path}")
                nested_dirs.append(str(item))
            else:
                print(f"跳过: {item} (无同名嵌套目录)")

    return nested_dirs

def main():
    # search_dir = "/opt/ros/jazzy/include"
    search_dir = "./install"

    print("=" * 50)
    print("查找同名嵌套目录工具")
    print("=" * 50)

    nested_dirs = find_nested_directories_detailed(search_dir)

    print("\n" + "=" * 50)
    if nested_dirs:
        print(f"找到 {len(nested_dirs)} 个包含同名嵌套目录的路径")
        print("输出格式适用于 .clangd 文件:")
        print()
        for dir_path in sorted(nested_dirs):
            print(f'    - "-I{dir_path}"')
    else:
        print("未找到同名嵌套目录")
    print("=" * 50)

if __name__ == "__main__":
    main()
