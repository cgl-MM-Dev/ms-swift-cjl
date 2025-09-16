#!/usr/bin/env python3
"""
将 YAML 配置文件转换为 Swift 命令行参数
"""

import yaml
import sys
import os

def yaml_to_swift_args(yaml_file):
    """将 YAML 配置转换为 Swift 命令行参数"""
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        args = []
        
        # 获取训练类型，用于判断是否需要处理LoRA配置
        train_type = config.get('train_type', 'lora')
        is_lora_training = train_type in ['lora', 'longlora', 'adalora', 'adapter', 'vera', 'boft', 'fourierft', 'reft']
        
        # 基础参数
        if 'tuner_backend' in config:
            args.extend(['--tuner_backend', config['tuner_backend']])
        if 'train_type' in config:
            args.extend(['--train_type', config['train_type']])
        if 'seed' in config:
            args.extend(['--seed', str(config['seed'])])
        
        # 模型参数
        if 'model' in config:
            args.extend(['--model', config['model']])
        if 'model_type' in config:
            args.extend(['--model_type', config['model_type']])
        if 'torch_dtype' in config:
            args.extend(['--torch_dtype', config['torch_dtype']])
        if 'attn_impl' in config:
            args.extend(['--attn_impl', config['attn_impl']])
        
        # 数据参数
        if 'dataset' in config:
            args.extend(['--dataset', config['dataset']])
        if 'split_dataset_ratio' in config:
            args.extend(['--split_dataset_ratio', str(config['split_dataset_ratio'])])
        if 'dataset_num_proc' in config:
            args.extend(['--dataset_num_proc', str(config['dataset_num_proc'])])
        if 'load_from_cache_file' in config:
            args.extend(['--load_from_cache_file', str(config['load_from_cache_file']).lower()])
        if 'dataset_shuffle' in config:
            args.extend(['--dataset_shuffle', str(config['dataset_shuffle']).lower()])
        if 'remove_unused_columns' in config:
            args.extend(['--remove_unused_columns', str(config['remove_unused_columns']).lower()])
        
        # 训练参数
        if 'num_train_epochs' in config:
            args.extend(['--num_train_epochs', str(config['num_train_epochs'])])
        if 'per_device_train_batch_size' in config:
            args.extend(['--per_device_train_batch_size', str(config['per_device_train_batch_size'])])
        if 'per_device_eval_batch_size' in config:
            args.extend(['--per_device_eval_batch_size', str(config['per_device_eval_batch_size'])])
        if 'learning_rate' in config:
            args.extend(['--learning_rate', str(config['learning_rate'])])
        if 'gradient_accumulation_steps' in config:
            args.extend(['--gradient_accumulation_steps', str(config['gradient_accumulation_steps'])])
        if 'warmup_ratio' in config:
            args.extend(['--warmup_ratio', str(config['warmup_ratio'])])
        if 'dataloader_num_workers' in config:
            args.extend(['--dataloader_num_workers', str(config['dataloader_num_workers'])])
        if 'optim' in config:
            args.extend(['--optim', config['optim']])
        if 'lr_scheduler_type' in config:
            args.extend(['--lr_scheduler_type', config['lr_scheduler_type']])
        if 'lr_scheduler_kwargs' in config:
            import json
            import shlex
            lr_scheduler_kwargs_str = json.dumps(config['lr_scheduler_kwargs'])
            # 使用 shlex.quote 正确转义 JSON 字符串
            args.extend(['--lr_scheduler_kwargs', shlex.quote(lr_scheduler_kwargs_str)])
        if 'weight_decay' in config:
            args.extend(['--weight_decay', str(config['weight_decay'])])
        if 'max_grad_norm' in config:
            args.extend(['--max_grad_norm', str(config['max_grad_norm'])])
        
        # LoRA 参数 - 只在LoRA训练时处理lora_configs块
        if is_lora_training and 'lora_configs' in config and config['lora_configs']:
            lora_config = config['lora_configs']
            if 'lora_rank' in lora_config:
                args.extend(['--lora_rank', str(lora_config['lora_rank'])])
            if 'lora_alpha' in lora_config:
                args.extend(['--lora_alpha', str(lora_config['lora_alpha'])])
            if 'lora_dropout' in lora_config:
                args.extend(['--lora_dropout', str(lora_config['lora_dropout'])])
            if 'target_modules' in lora_config:
                args.extend(['--target_modules', lora_config['target_modules']])
            if 'use_dora' in lora_config:
                args.extend(['--use_dora', str(lora_config['use_dora']).lower()])
            if 'use_rslora' in lora_config:
                args.extend(['--use_rslora', str(lora_config['use_rslora']).lower()])
            if 'lorap_lr_ratio' in lora_config and lora_config['lorap_lr_ratio'] is not None:
                args.extend(['--lorap_lr_ratio', str(lora_config['lorap_lr_ratio'])])
        
        # 多模态参数
        if 'max_length' in config:
            args.extend(['--max_length', str(config['max_length'])])
        if 'max_pixels' in config:
            args.extend(['--max_pixels', str(config['max_pixels'])])
        if 'freeze_vit' in config:
            args.extend(['--freeze_vit', str(config['freeze_vit']).lower()])
        if 'padding_free' in config:
            args.extend(['--padding_free', str(config['padding_free']).lower()])
        
        # 模板参数
        if 'template' in config and config['template'] is not None:
            args.extend(['--template', config['template']])
        if 'system' in config and config['system'] is not None:
            args.extend(['--system', config['system']])
        if 'truncation_strategy' in config:
            args.extend(['--truncation_strategy', config['truncation_strategy']])
        if 'use_chat_template' in config:
            args.extend(['--use_chat_template', str(config['use_chat_template']).lower()])
        if 'template_backend' in config:
            args.extend(['--template_backend', config['template_backend']])
        
        # 损失函数 - 跳过 default 值，使用 Swift 默认
        if 'loss_type' in config and config['loss_type'] != 'default':
            args.extend(['--loss_type', config['loss_type']])
        
        # 输出和日志
        if 'output_dir' in config:
            args.extend(['--output_dir', config['output_dir']])
        if 'logging_steps' in config:
            args.extend(['--logging_steps', str(config['logging_steps'])])
        if 'eval_steps' in config:
            args.extend(['--eval_steps', str(config['eval_steps'])])
        if 'save_strategy' in config:
            args.extend(['--save_strategy', config['save_strategy']])
        if 'save_steps' in config:
            args.extend(['--save_steps', str(config['save_steps'])])
        if 'save_only_model' in config:
            args.extend(['--save_only_model', str(config['save_only_model']).lower()])
        if 'save_total_limit' in config:
            args.extend(['--save_total_limit', str(config['save_total_limit'])])
        if 'report_to' in config:
            args.extend(['--report_to', config['report_to']])
        
        # Wandb 配置 - Swift 支持 run_name 参数
        if 'wandb_run_name' in config:
            args.extend(['--run_name', config['wandb_run_name']])
        # wandb_project 和 wandb_tags 通过环境变量设置
        
        # DeepSpeed
        if 'deepspeed' in config:
            args.extend(['--deepspeed', config['deepspeed']])
        
        # 全参数训练相关参数
        if 'freeze_parameters' in config and config['freeze_parameters']:
            if isinstance(config['freeze_parameters'], list):
                for param in config['freeze_parameters']:
                    args.extend(['--freeze_parameters', param])
            else:
                args.extend(['--freeze_parameters', config['freeze_parameters']])
        
        if 'freeze_parameters_regex' in config and config['freeze_parameters_regex'] is not None:
            args.extend(['--freeze_parameters_regex', config['freeze_parameters_regex']])
        
        if 'trainable_parameters' in config and config['trainable_parameters']:
            if isinstance(config['trainable_parameters'], list):
                for param in config['trainable_parameters']:
                    args.extend(['--trainable_parameters', param])
            else:
                args.extend(['--trainable_parameters', config['trainable_parameters']])
        
        # model_kwargs - DeepSpeed 模式下需要特殊处理
        if 'model_kwargs' in config and config['model_kwargs']:
            import json
            model_kwargs_dict = config['model_kwargs'].copy()
            
            # DeepSpeed 不兼容 device_map，移除相关设置
            if 'deepspeed' in config and config['deepspeed']:
                # DeepSpeed 会自动处理设备分配，不需要 device_map
                model_kwargs_dict.pop('device_map', None)
                
                # DeepSpeed + torchrun 下避免 JSON 传递问题
                # 改为使用 --max_pixels 单独参数
                # if 'max_pixels' in model_kwargs_dict:
                #     args.extend(['--max_pixels', str(model_kwargs_dict['max_pixels'])])
                #     model_kwargs_dict.pop('max_pixels', None)
            
            # 如果还有其他参数，才使用 JSON 方式
            if model_kwargs_dict:
                model_kwargs_str = json.dumps(model_kwargs_dict)
                # 使用双重转义避免 torchrun 问题
                args.extend(['--model_kwargs', model_kwargs_str.replace(" ", "")])        
        return args
        
    except Exception as e:
        print(f"# Error parsing YAML file: {e}", file=sys.stderr)
        return []

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python yaml_to_args.py <yaml_file>", file=sys.stderr)
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    if not os.path.exists(yaml_file):
        print(f"# Error: YAML file {yaml_file} not found", file=sys.stderr)
        sys.exit(1)
    
    args = yaml_to_swift_args(yaml_file)
    print(' '.join(args))
