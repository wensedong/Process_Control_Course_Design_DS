#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PID参数整定程序 — 无控制台启动器
双击此文件即可启动 GUI，不会弹出黑色命令行窗口。
"""
import os
import sys

# 确保工作目录正确
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from app import main

if __name__ == '__main__':
    main()
