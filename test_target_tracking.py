#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标跟踪功能测试示例
演示如何使用新的目标跟踪器实现目标确认和持久化显示
"""

import time
import numpy as np
import cv2
from rknn_image import TargetTracker, CLASSES_CHINESE, add_detection_list_to_image

def simulate_detection():
    """模拟目标检测过程"""
    
    # 创建目标跟踪器
    tracker = TargetTracker(confirmation_window=5, min_confirmations=3, persistence_timeout=10.0)
    
    # 模拟图像
    height, width = 480, 640
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 模拟检测结果序列
    # 场景：一个手机目标在不同帧中时隐时现
    simulation_frames = [
        # 帧1：检测到手机
        [(100, 100, 200, 200, "手机", 0.85)],
        # 帧2：检测到手机
        [(105, 105, 205, 205, "手机", 0.87)],
        # 帧3：未检测到手机
        [],
        # 帧4：检测到手机
        [(110, 110, 210, 210, "手机", 0.89)],
        # 帧5：检测到手机
        [(108, 108, 208, 208, "手机", 0.90)],
        # 帧6：未检测到手机
        [],
        # 帧7：未检测到手机
        [],
        # 帧8：检测到手机（应该已经确认为持久目标）
        [(115, 115, 215, 215, "手机", 0.88)],
        # 帧9：未检测到手机（持久目标应该仍然显示）
        [],
        # 帧10：未检测到手机（持久目标应该仍然显示）
        []
    ]
    
    print("开始目标跟踪测试...")
    print("=" * 50)
    
    for frame_idx, detections in enumerate(simulation_frames):
        current_time = frame_idx * 1.0  # 每帧间隔1秒
        
        # 准备检测数据
        detection_data = []
        for det in detections:
            box = det[0:4]
            class_name = det[4]
            score = det[5]
            detection_data.append((box, class_name, score))
        
        # 更新跟踪器
        persistent_targets = tracker.update(detection_data, current_time)
        
        # 清空图像
        image.fill(0)
        
        # 绘制所有目标
        all_targets = tracker.get_all_targets()
        
        # 绘制已确认的目标（蓝色）
        for target in persistent_targets:
            box = target['bbox']
            class_name = target['classes']
            score = target['score']
            
            # 绘制边界框
            cv2.rectangle(image, 
                         (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), 
                         (255, 0, 0), 2)
            # 绘制标签
            cv2.putText(image, f"{class_name} {score:.2f}", 
                        (int(box[0]), int(box[1]) - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 绘制未确认的目标（绿色）
        for target_id, target_info in all_targets.items():
            if not target_info['confirmed']:
                box = target_info['bbox']
                class_name = target_info['class_name']
                score = target_info['score']
                
                # 绘制边界框
                cv2.rectangle(image, 
                             (int(box[0]), int(box[1])), 
                             (int(box[2]), int(box[3])), 
                             (0, 255, 0), 1)
                # 绘制标签
                cv2.putText(image, f"{class_name}?", 
                            (int(box[0]), int(box[1]) - 6), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 添加右侧显示
        confirmed_list = [{"classes": target['classes'], "score": target['score'], "last_seen": target['last_seen']} 
                         for target in persistent_targets]
        image = add_detection_list_to_image(image=image, detection_list=confirmed_list)
        
        # 显示帧信息
        cv2.putText(image, f"帧: {frame_idx + 1}/10", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 打印状态信息
        print(f"帧 {frame_idx + 1}: 检测到 {len(detection_data)} 个目标")
        print(f"  - 已确认目标: {len(persistent_targets)} 个")
        print(f"  - 总计跟踪目标: {len(all_targets)} 个")
        
        if persistent_targets:
            for target in persistent_targets:
                time_since_seen = current_time - target['last_seen']
                print(f"    * {target['classes']} (置信度: {target['score']:.2f}, 最后检测: {time_since_seen:.1f}s前)")
        
        print("-" * 30)
        
        # 显示图像
        cv2.imshow(f"目标跟踪测试 - 帧 {frame_idx + 1}", image)
        cv2.waitKey(1000)  # 等待1秒
    
    cv2.destroyAllWindows()
    print("测试完成！")
    
    # 总结
    print("\n测试总结:")
    print(f"- 目标确认窗口: {tracker.confirmation_window} 帧")
    print(f"- 最小确认次数: {tracker.min_confirmations} 次")
    print(f"- 持久化超时时间: {tracker.persistence_timeout} 秒")
    print(f"- 最终确认的目标数: {len(tracker.get_persistent_targets())}")

if __name__ == "__main__":
    simulate_detection()