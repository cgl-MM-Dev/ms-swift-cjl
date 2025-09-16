#!/usr/bin/env python3
"""
从 YAML 配置文件中解析 Wandb 配置并输出为 bash 可执行的格式
"""

import yaml
import sys
import os

def parse_wandb_config(yaml_file):
    """解析 YAML 中的 Wandb 配置"""
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Wandb 项目名
        if 'wandb_project' in config:
            print(f'export WANDB_PROJECT="{config["wandb_project"]}"')
        
        # Wandb 运行名称 - 通过 --run_name 参数设置，不使用环境变量
        # if 'wandb_run_name' in config:
        #     print(f'export WANDB_RUN_NAME="{config["wandb_run_name"]}"')
        
        # Wandb 标签
        if 'wandb_tags' in config and config['wandb_tags']:
            tags_str = ','.join(config['wandb_tags'])
            print(f'export WANDB_TAGS="{tags_str}"')
            
    except Exception as e:
        print(f"# Error parsing YAML file: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_wandb_config.py <yaml_file>", file=sys.stderr)
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    if not os.path.exists(yaml_file):
        print(f"# Error: YAML file {yaml_file} not found", file=sys.stderr)
        sys.exit(1)
    
    parse_wandb_config(yaml_file)
