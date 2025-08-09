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
  - 使用PIL库绘制中文文本以支持中文显示

- **detect.py**: GUI应用程序，提供可视化界面进行目标检测
  - `SimpleDetector`: 简单单线程检测器
  - `DetectionApp`: 主应用程序类，管理UI和检测流程
  - 使用PIL库绘制中文文本以支持中文显示

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

## 中文显示支持

项目使用PIL库来支持中文文本显示：
- 在detect.py中，`draw_detection_results`和`create_detection_list_image`函数使用PIL绘制中文文本
- 在rknn_image.py中，`add_detection_list_to_image`函数使用PIL绘制中文文本
- 在func.py中，`create_detection_list_image`函数使用PIL绘制中文文本
- 需要系统中安装中文字体（如simhei.ttf）
- 如果字体文件不存在，程序会自动回退到默认字体

## 更换检测模型

如果需要更换检测模型，需要更新以下标签对应关系：

1. **rknn_image.py文件**：
   - `CLASSES`元组：包含模型输出的英文类别标签
   - `CLASSES_CHINESE`字典：包含英文类别标签到中文显示标签的映射

2. **func.py文件**：
   - `CLASSES`元组：需要与rknn_image.py中的CLASSES保持一致

3. **detect.py文件**：
   - `CLASSES`元组：需要与rknn_image.py中的CLASSES保持一致
   - 中文到英文的映射字典：在`process_detections`函数中的`class_map`字典

## 开发注意事项

- 修改推理逻辑时主要编辑`func.py`中的`myFunc`函数
- 目标跟踪参数可在`rknn_image.py`中的`TargetTracker`类中调整
- 多线程数量需要根据硬件散热能力调整，避免过热
- NPU核心分配逻辑在`rknnpool.py`的`initRKNN()`函数中
- 中文显示依赖PIL库和系统字体文件
- 图像布局逻辑在多个文件中实现，需要保持一致性