# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个基于RKNN的多线程目标检测项目，专为RK3588/RK3588s等芯片优化。项目使用多线程异步操作RKNN模型来提高NPU使用率和推理帧数。

## 核心架构

- **rknnpool.py**: 核心线程池管理器，负责初始化多个RKNN实例并管理线程池执行
  - `rknnPoolExecutor`: 主要的池执行器类，管理RKNN模型的多线程推理
  - `initRKNN()`: 初始化单个RKNN实例，根据ID分配到不同的NPU核心
  - `initRKNNs()`: 批量初始化多个RKNN实例

- **func.py**: 推理函数和后处理逻辑
  - `myFunc()`: 主要推理函数，处理单帧图像并返回检测结果
  - YOLO后处理函数：`process()`, `filter_boxes()`, `nms_boxes()`, `yolov5_post_process()`

- **rknn_image.py**: 图像处理和目标跟踪
  - `TargetTracker`: 目标跟踪器类，实现目标确认、持久化和超时管理
  - 包含滑动窗口确认机制，避免检测抖动

- **main.py**: 主执行文件，设置摄像头/视频输入，初始化线程池并运行推理循环

## 常用命令

### 运行项目
```bash
python main.py
```

### 性能调优（需要root权限）
```bash
# 开启性能模式（CPU定频）
sudo bash performance.sh

# 监控NPU占用和温度
bash rkcat.sh
```

### 目标跟踪测试
```bash
python test_target_tracking.py
```

## 配置说明

### main.py中的关键配置：
- `modelPath`: RKNN模型路径，默认使用yolo11l_i8_train.rknn
- `TPEs`: 线程数量，建议1-3个线程（过多会导致温度过高）
- `cap`: 视频源配置，支持视频文件或摄像头

### 性能调优建议：
- 1线程：温度约56°
- 2线程：温度约64° 
- 3线程：温度约69°
- 超过3线程温度会显著增加，需要充分散热

## 模型支持

项目支持多种YOLO模型：
- yolo11l_fp_train.rknn / yolo11l_i8_train.rknn
- yolo11m_fp_train.rknn / yolo11m_i8_train.rknn  
- yolov5s_relu_tk2_RK3588_i8.rknn

模型文件应放置在`rknnModel/`目录下。

## 目标跟踪特性

- **滑动窗口确认**: 使用10帧窗口，最少3次确认来稳定检测
- **目标持久化**: 确认的目标在短暂消失后仍会显示
- **IOU匹配**: 基于IOU和距离的目标匹配算法
- **超时管理**: 10秒无检测后自动清理目标

## 开发注意事项

- 修改推理逻辑时主要编辑`func.py`中的`myFunc`函数
- 目标跟踪参数可在`rknn_image.py`中的`TargetTracker`类中调整
- 多线程数量需要根据硬件散热能力调整，避免过热
- NPU核心分配逻辑在`rknnpool.py`的`initRKNN()`函数中