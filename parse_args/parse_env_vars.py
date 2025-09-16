#!/usr/bin/env python3
"""
从 YAML 配置文件中解析环境变量并输出为 bash 可执行的格式
"""

import yaml
import sys
import os

def parse_env_vars(yaml_file):
    """解析 YAML 文件中的环境变量"""
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        env_vars = config.get('env_vars', {})
        
        # 输出 bash export 命令
        for key, value in env_vars.items():
            print(f'export {key}="{value}"')
            
    except Exception as e:
        print(f"# Error parsing YAML file: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_env_vars.py <yaml_file>", file=sys.stderr)
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    if not os.path.exists(yaml_file):
        print(f"# Error: YAML file {yaml_file} not found", file=sys.stderr)
        sys.exit(1)
    
    parse_env_vars(yaml_file)
