import urllib
import time
import sys
import numpy as np
import cv2
from rknnlite.api import RKNNLite
import os
import cv2
import argparse
from collections import defaultdict, deque
# add path
# realpath = os.path.abspath(__file__)
# _sep = os.path.sep
# realpath = realpath.split(_sep)
# sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

from coco_utils import COCO_test_helper
import numpy as np
from PIL import Image, ImageDraw, ImageFont
# RKNN_MODEL = '/home/cat/programs/lubancat-flask-opencv-rknn/controller/utils/yolo11m_i8_train.rknn'
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)
from multiprocessing import Manager
manager = Manager()
from collections import defaultdict
detection_history = defaultdict(list)  # 格式: {"目标名": [帧1结果, 帧2结果...]}
FRAME_WINDOW_SIZE = 10  # 滑动窗口大小（帧数）
MIN_DETECT_COUNT = 3   # 最小确认次数
confirmed_targets = manager.dict()  # 格式: {"目标ID": {"classes": "person", "last_seen": timestamp}}
TARGET_TIMEOUT = 10.0


# 简单的ID计数器
next_target_id = 1

class TargetTracker:
    """目标跟踪器类，实现目标的确认、持久化和超时管理"""
    
    def __init__(self, confirmation_window=10, min_confirmations=3, persistence_timeout=10.0, iou_threshold=0.3):
        self.confirmation_window = confirmation_window  # 确认窗口大小（帧数）
        self.min_confirmations = min_confirmations     # 最小确认次数
        self.persistence_timeout = persistence_timeout  # 持久化超时时间（秒）
        self.iou_threshold = iou_threshold             # IOU阈值，用于目标匹配
        
        # 使用多进程安全的字典
        self.tracked_targets = manager.dict()  # 格式: {target_id: 目标信息}
        self.detection_history = defaultdict(list)  # 检测历史记录
        
    def calculate_iou(self, box1, box2):
        """计算两个边界框的IOU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_distance(self, box1, box2):
        """计算两个边界框中心点的欧式距离"""
        center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        return ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
    
    def find_best_match(self, detected_box, class_name, current_time):
        """为检测到的目标找到最佳的匹配"""
        best_match_id = None
        best_score = 0.0
        
        for target_id, target_info in self.tracked_targets.items():
            if target_info['class_name'] != class_name:
                continue
                
            # 计算匹配分数（结合IOU和距离）
            iou_score = self.calculate_iou(detected_box, target_info['bbox'])
            distance_score = 1.0 / (1.0 + self.calculate_distance(detected_box, target_info['bbox']))
            
            # 综合评分
            total_score = 0.7 * iou_score + 0.3 * distance_score
            
            if total_score > best_score and total_score > self.iou_threshold:
                best_score = total_score
                best_match_id = target_id
        
        return best_match_id
    
    def generate_target_id(self, class_name, box):
        """基于坐标生成稳定的目标ID"""
        # 计算边界框中心点
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        # 直接使用中心坐标作为ID，确保位置相关性
        return f"{class_name}_{int(center_x):03d}_{int(center_y):03d}"
    
    def update(self, detections, current_time):
        """更新跟踪器状态 - 基于锚点位置的目标确认策略
        
        Args:
            detections: 当前帧的检测结果 [(box, class_name, score), ...]
            current_time: 当前时间戳
        """
        # 已处理的检测结果索引，避免一个检测被多个目标匹配
        processed_detections = set()
        
        # 1. 更新现有目标：在IOU范围内的检测归并到现有目标
        for target_id, target_info in list(self.tracked_targets.items()):
            if not target_info.get('confirmed', False):
                continue
                
            best_detection_idx = None
            best_score = 0.0
            
            # 找到与当前目标最匹配的检测结果
            for i, (box, class_name, score) in enumerate(detections):
                if i in processed_detections or class_name != target_info['class_name']:
                    continue
                    
                iou_score = self.calculate_iou(box, target_info['bbox'])
                if iou_score > self.iou_threshold and iou_score > best_score:
                    best_score = iou_score
                    best_detection_idx = i
            
            # 如果找到匹配的检测，更新目标信息
            if best_detection_idx is not None:
                box, class_name, score = detections[best_detection_idx]
                processed_detections.add(best_detection_idx)
                
                # 更新目标信息，但保持原始锚点ID不变
                target_info_copy = dict(target_info)
                target_info_copy.update({
                    'bbox': box,
                    'score': score,
                    'last_seen': current_time
                })
                self.tracked_targets[target_id] = target_info_copy
                
                # 更新检测历史
                if target_id not in self.detection_history:
                    self.detection_history[target_id] = deque(maxlen=self.confirmation_window)
                self.detection_history[target_id].append(current_time)
        
        # 2. 处理未匹配的检测：创建新的锚点目标（但需要多帧验证）
        for i, (box, class_name, score) in enumerate(detections):
            if i in processed_detections:
                continue
                
            # 基于首次检测位置生成锚点ID
            anchor_id = self.generate_target_id(class_name, box)
            
            # 检查是否已存在相同锚点ID的目标（可能是之前未确认的）
            if anchor_id in self.tracked_targets:
                # 更新现有未确认目标
                target_info = dict(self.tracked_targets[anchor_id])
                target_info.update({
                    'bbox': box,
                    'score': score,
                    'last_seen': current_time,
                    'detection_count': target_info.get('detection_count', 0) + 1
                })
                self.tracked_targets[anchor_id] = target_info
            else:
                # 创建新的锚点目标，初始为未确认状态
                self.tracked_targets[anchor_id] = {
                    'class_name': class_name,
                    'bbox': box,
                    'score': score,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'confirmed': False,  # 需要多帧验证
                    'confirmed_at': None,
                    'detection_count': 1
                }
            
            # 更新检测历史
            if anchor_id not in self.detection_history:
                self.detection_history[anchor_id] = deque(maxlen=self.confirmation_window)
            self.detection_history[anchor_id].append(current_time)
            
            # 检查确认条件：多帧验证
            target_info = self.tracked_targets[anchor_id]
            if not target_info['confirmed']:
                detection_count = len(self.detection_history[anchor_id])
                if detection_count >= self.min_confirmations:
                    # 达到确认条件，设为已确认
                    target_info_copy = dict(target_info)
                    target_info_copy.update({
                        'confirmed': True,
                        'confirmed_at': current_time
                    })
                    self.tracked_targets[anchor_id] = target_info_copy
        
        # 3. 清理超时目标
        self.cleanup_timeout_targets(current_time)
        
        # 4. 返回已确认的目标列表
        return self.get_persistent_targets()
    
    def cleanup_timeout_targets(self, current_time):
        """清理超时的目标"""
        targets_to_remove = []
        
        for target_id, target_info in self.tracked_targets.items():
            # 未确认的目标超时时间较短
            timeout = self.persistence_timeout * 0.3 if not target_info['confirmed'] else self.persistence_timeout
            
            if current_time - target_info['last_seen'] > timeout:
                targets_to_remove.append(target_id)
        
        for target_id in targets_to_remove:
            # 从跟踪字典中移除
            del self.tracked_targets[target_id]
            if target_id in self.detection_history:
                del self.detection_history[target_id]
    
    def get_persistent_targets(self):
        """获取需要持久化显示的目标列表"""
        persistent_targets = []
        
        # print(f"DEBUG: tracked_targets count: {len(self.tracked_targets)}")
        for target_id, target_info in self.tracked_targets.items():
            # print(f"DEBUG: target {target_id}, confirmed: {target_info.get('confirmed', False)}")
            if target_info['confirmed']:
                persistent_targets.append({
                    'id': target_id,
                    'classes': target_info['class_name'],
                    'bbox': target_info['bbox'],
                    'score': target_info['score'],
                    'last_seen': target_info['last_seen']
                })
        
        # print(f"DEBUG: persistent_targets count: {len(persistent_targets)}")
        return persistent_targets
    
    def get_all_targets(self):
        """获取所有目标（包括已确认和未确认的）"""
        return dict(self.tracked_targets)


# 全局跟踪器实例 - 缩短超时时间
target_tracker = TargetTracker(confirmation_window=10, min_confirmations=5, persistence_timeout=1.0, iou_threshold=0.2) 

# The follew two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

# CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
#            "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
#            "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
#            "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
#            "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
#            "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop    ","mouse    ","remote ","keyboard ","cell phone","microwave ",
#            "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")
"""
protective_bag
safety_helmet
brush
mobile_phone
protective_flag
whistle
loudhailer
pliers
screwdriver
multimeter
backpack
walkie-talkie
toolbox
spotlight
wrench
medicine_box
glove
flashlight
"""
CLASSES = ("protective_bag",
           "safety_helmet",
           "brush",
           "mobile_phone",
           "protective_flag",
           "whistle",
           "loudhailer",
           "pliers",
           "screwdriver",
           "multimeter",
           "backpack",
           "walkie-talkie",
           "toolbox",
           "spotlight",
           "wrench",
           "medicine_box",
           "glove",
           "flashlight"
           )
CLASSES_CHINESE = {
    "protective_bag":   "防护袋",
    "safety_helmet":    "安全帽",
    "brush":            "刷子",
    "mobile_phone":     "手机",
    "protective_flag":  "防护旗",
    "whistle":          "哨子",
    "loudhailer":       "扩音器",
    "pliers":           "钳子",
    "screwdriver":      "螺丝刀",
    "multimeter":       "万用表",
    "backpack":         "背包",
    "walkie-talkie":    "对讲机",
    "toolbox":          "工具箱",
    "spotlight":        "探照灯",
    "wrench":           "扳手",
    "medicine_box":     "医药箱",
    "glove":            "手套",
    "flashlight":       "手电筒"
}
coco_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18]

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def dfl(position):
    # Distribution Focal Loss (DFL)
    """
    Distribution Focal Loss (DFL) - NumPy 实现
    Args:
        position: 输入张量，形状 (n, c, h, w)
    Returns:
        y: 输出张量，形状 (n, p_num, h, w)
    """
    x = np.array(position)
    n, c, h, w = x.shape
    p_num = 4  
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w)
    y_exp = np.exp(y - np.max(y, axis=2, keepdims=True))
    y_softmax = y_exp / np.sum(y_exp, axis=2, keepdims=True)
    acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(np.float32)
    y = np.sum(y_softmax * acc_metrix, axis=2)
 
    return y
    import torch
    x = torch.tensor(position)
    n,c,h,w = x.shape
    p_num = 4
    mc = c//p_num
    y = x.reshape(n,p_num,mc,h,w)
    y = y.softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y.numpy()
 

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def draw(image, boxes, scores, classes):
    # for box, score, cl in zip(boxes, scores, classes):
    #     top, left, right, bottom = [int(_b) for _b in box]
    #     # print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
    results = []
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        target_id = f"{CLASSES[cl]}_{int(top)}_{int(left)}"  # 用坐标生成唯一ID
        results.append({"id": target_id, "classes": CLASSES[cl]})
        
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
        platform = 'pytorch'
        from py_utils.pytorch_executor import Torch_model_container
        model = Torch_model_container(args.model_path)
    elif model_path.endswith('.rknn'):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container 
        model = RKNN_model_container(args.model_path, args.target, args.device_id)
    elif model_path.endswith('onnx'):
        platform = 'onnx'
        from py_utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(args.model_path)
    else:
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform

def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False
def find_closest_target(box, confirmed_targets, threshold=50):
    center = ((box[0] + box[2])//2, (box[1] + box[3])//2)
    for target_id, info in confirmed_targets.items():
        if not info["confirmed"]:
            prev_box = info["bbox"]
            prev_center = ((prev_box[0] + prev_box[2])//2, (prev_box[1] + prev_box[3])//2)
            distance = ((center[0]-prev_center[0])**2 + (center[1]-prev_center[1])**2)**0.5
            if distance < threshold:
                return target_id
    return None

def process_image(image, outputs, coco_helper, overlay_mode=True, add_detection_list=True):
    """
    处理图像并更新目标跟踪器
    
    Args:
        image: 输入图像
        outputs: 模型输出
        coco_helper: COCO辅助对象
    Returns:
        image: 处理后的图像（包含边界框和右侧标签）
        confirmed_list: 已确认的目标列表
    """
    current_time = time.time()
    global target_tracker
    
    # 1. 处理当前帧检测结果
    boxes, classes, scores = post_process(outputs)
    detections = []  # 用于跟踪器的检测结果列表

    if boxes is not None:
        real_boxes = coco_helper.get_real_box(boxes)  # 获取调整后的bbox
        for box, cl, score in zip(real_boxes, classes, scores):
            class_name = CLASSES[cl]  # 使用英文类名
            detections.append((box, class_name, score))

    # 2. 更新目标跟踪器
    persistent_targets = target_tracker.update(detections, current_time)
    
    # 3. 获取所有目标（包括未确认的）用于调试和显示
    all_targets = target_tracker.get_all_targets()

    # 4. 绘制目标
    # 4.1 绘制已确认的目标（蓝色边界框）
    used_label_positions = []  # 记录已使用的标签位置，避免重叠
    
    for target in persistent_targets:
        box = target['bbox']
        class_name = target['classes']
        score = target['score']
        target_id = target['id']
        
        # 生成简短的显示ID（取target_id的最后4位数字）
        display_id = target_id.split('_')[-1][-4:]
        
        # 绘制蓝色边界框表示已确认的目标
        cv2.rectangle(image, 
                     (int(box[0]), int(box[1])), 
                     (int(box[2]), int(box[3])), 
                     (255, 0, 0), 2)  # 蓝色
        
        # 计算标签位置，避免重叠
        label = f"ID{display_id} {class_name} {score:.2f}"
        label_x = int(box[0])
        label_y = int(box[1]) - 6
        
        # 检查并调整标签位置避免重叠
        for used_pos in used_label_positions:
            if abs(label_x - used_pos[0]) < 150 and abs(label_y - used_pos[1]) < 20:
                label_y = used_pos[1] - 25  # 向上偏移
        
        # 确保标签不会超出图像边界
        if label_y < 20:
            label_y = int(box[3]) + 20  # 移到框下方
        
        # 记录当前标签位置
        used_label_positions.append((label_x, label_y))
        
        # 绘制标签
        cv2.putText(image, label, 
                    (label_x, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 4.2 绘制未确认的目标（绿色边界框，可选）
    for target_id, target_info in all_targets.items():
        if not target_info['confirmed']:
            box = target_info['bbox']
            class_name = target_info['class_name']
            score = target_info['score']
            
            # 绘制绿色边界框表示未确认的目标
            cv2.rectangle(image, 
                         (int(box[0]), int(box[1])), 
                         (int(box[2]), int(box[3])), 
                         (0, 255, 0), 1)  # 绿色，细线
            # 绘制标签 - 使用英文标签
            cv2.putText(image, f"{class_name}?", 
                        (int(box[0]), int(box[1]) - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 5. 生成右侧标签列表 - 使用跟踪器的持久化目标，添加ID信息和边界框
    confirmed_list = []
    current_time_threshold = current_time - 0.5  # 当前帧的时间阈值（0.5秒内）
    
    for target in persistent_targets:
        target_id = target['id']
        display_id = target_id.split('_')[-1][-4:]  # 生成简短显示ID
        
        # 判断是否为当前帧的目标
        is_current_frame = target['last_seen'] >= current_time_threshold
        
        confirmed_list.append({
            "classes": CLASSES_CHINESE.get(target['classes'], target['classes']),
            "score": target['score'],
            "last_seen": target['last_seen'],
            "display_id": display_id,  # 添加显示ID
            "is_current_frame": is_current_frame,  # 标记是否为当前帧
            "bbox": target['bbox']  # 添加边界框信息
        })
    
    # 如果跟踪器没有目标，则显示当前帧检测结果作为备选
    if not confirmed_list and boxes is not None:
        real_boxes = coco_helper.get_real_box(boxes)
        for i, (box, cl, score) in enumerate(zip(real_boxes, classes, scores)):
            class_name = CLASSES[cl]
            chinese_name = CLASSES_CHINESE.get(class_name, class_name)
            confirmed_list.append({
                "classes": chinese_name,
                "score": score,
                "last_seen": current_time,
                "display_id": f"{i+1:04d}",  # 临时ID
                "is_current_frame": True
            })
    
    # 只在需要时添加检测列表到图像右侧
    # 使用覆盖模式将检测结果覆盖在原始图像上，而不是拼接
    if add_detection_list:
        image = add_detection_list_to_image(image=image, detection_list=confirmed_list, overlay_mode=overlay_mode)

    return image, confirmed_list if confirmed_list else None
        
def add_detection_list_to_image(image, detection_list, add_width=480,
                               background_color=(30, 30, 30),   # 深灰色背景
                               title_color=(255, 255, 255),    # 白色标题
                               text_color=(200, 200, 200),     # 浅灰色文字
                               highlight_color=(255, 255, 0),  # 黄色高亮
                               title="检测目标列表",
                               overlay_mode=False):
    """
    在图像右侧添加纯色背景和持久化检测对象列表，或者以覆盖模式显示
    
    参数:
        image: 输入图像
        detection_list: 检测对象列表，例如 [{'classes': 'person', 'score': 0.85}, ...]
        add_width: 右侧扩展宽度
        background_color: 背景颜色 (B, G, R)
        title_color: 标题颜色 (B, G, R)
        text_color: 文字颜色 (B, G, R)
        highlight_color: 高亮颜色 (B, G, R)
        title: 标题文本
        overlay_mode: 是否使用覆盖模式（True: 覆盖在原始图像上, False: 拼接在右侧）
    """
    import time
    h, w = image.shape[:2]
    
    if overlay_mode:
        # 覆盖模式：在原始图像的半透明背景上显示信息
        # 创建一个半透明的覆盖层
        overlay = image.copy()
        cv2.rectangle(overlay, (w - add_width, 0), (w, h), background_color, -1)
        # 添加半透明效果
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        new_img = image
        display_w = w  # 显示宽度为原始图像宽度
    else:
        # 拼接模式：在图像右侧添加区域
        # 创建新图像
        new_w = w + add_width
        new_h = h
        new_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        new_img[:, :w] = image
        new_img[:, w:] = background_color
        display_w = new_w  # 显示宽度为原始图像+扩展区域
    
    # 设置字体
    cv2_font = cv2.FONT_HERSHEY_SIMPLEX
    
    if not detection_list:
        # 显示"暂无确认目标"
        if overlay_mode:
            # 覆盖模式下在图像右侧绘制
            text_x = w - add_width + 20
            text_y = h // 2 - 20
        else:
            # 拼接模式下在扩展区域绘制
            text_x = w + 20
            text_y = h // 2 - 20
            
        img_pil = Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("/home/cat/programs/rknn-multi-threaded/simhei.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        text = "暂无确认目标"
        draw.text((text_x, text_y), text, font=font, fill=tuple(title_color[::-1]))
        new_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return new_img
    
    # 计算布局 - 使用表格式布局
    title_height = 60
    item_height = 35  # 减小行高以容纳更多项目
    margin = 20
    padding = 15
    
    # 根据模式计算可用空间
    if overlay_mode:
        # 覆盖模式下，可用宽度是覆盖区域的宽度
        available_width = add_width - 2 * margin
        # 可用高度是整个图像高度
        available_height = h - padding - title_height - 60  # 预留底部空间
        # 起始x坐标在覆盖区域内
        start_x = w - add_width
    else:
        # 拼接模式下，可用宽度是扩展区域的宽度
        available_width = add_width - 2 * margin
        # 可用高度是整个图像高度
        available_height = h - padding - title_height - 60  # 预留底部空间
        # 起始x坐标在原始图像右侧
        start_x = w
    
    # 每个格子的固定尺寸
    cell_width = 200  # 每个格子宽度
    cell_height = item_height
    
    # 计算最大列数和行数
    max_columns = max(1, available_width // cell_width)
    max_rows = max(1, available_height // cell_height)
    max_items_visible = max_columns * max_rows
    
    # 如果项目超出可显示范围，调整布局
    total_items = len(detection_list)
    if total_items > max_items_visible:
        # 重新计算以适应所有项目
        max_rows = max(1, (total_items + max_columns - 1) // max_columns)
        # 如果行数过多，增加列数
        if max_rows * cell_height > available_height:
            max_columns = min(available_width // 180, max(1, (total_items + 10 - 1) // 10))  # 最多10行
            max_rows = max(1, (total_items + max_columns - 1) // max_columns)
            cell_width = available_width // max_columns
    
    # 绘制标题
    img_pil = Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        title_font = ImageFont.truetype("/home/cat/programs/rknn-multi-threaded/simhei.ttf", 32)
        item_font = ImageFont.truetype("/home/cat/programs/rknn-multi-threaded/simhei.ttf", 24)
    except:
        title_font = ImageFont.load_default()
        item_font = ImageFont.load_default()
    
    # 绘制标题 - 显示当前帧的确认目标数量
    current_frame_confirmed = len([item for item in detection_list 
                                 if item.get('is_current_frame', False)])
    title_with_count = f"{title} 核准{current_frame_confirmed}个"
    draw.text((start_x + margin, padding), title_with_count, font=title_font, fill=tuple(title_color[::-1]))
    
    # 绘制分隔线
    draw.line([(start_x + margin, padding + title_height - 10), 
               (start_x + add_width - margin, padding + title_height - 10)], 
              fill=tuple(text_color[::-1]), width=2)
    
    # 显示目标列表 - 使用表格式布局
    start_y = padding + title_height
    
    for index, item in enumerate(detection_list):
        # 计算当前项目在表格中的位置
        row = index // max_columns
        col = index % max_columns
        
        # 计算格子的实际像素位置
        cell_x = start_x + margin + col * cell_width
        cell_y = start_y + row * cell_height
        
        # 跳过超出可显示区域的项目
        if cell_y + cell_height > h - 60:  # 预留底部空间
            break
        
        # 计算时间差
        time_since_last_seen = time.time() - item.get('last_seen', time.time())
        is_recent = time_since_last_seen < 2.0  # 最近2秒检测到的目标高亮显示
        
        # 选择颜色
        text_color_to_use = tuple(highlight_color[::-1]) if is_recent else tuple(text_color[::-1])
        
        # 绘制格子背景（可选，用于调试布局）
        # draw.rectangle([cell_x, cell_y, cell_x + cell_width - 5, cell_y + cell_height - 2], 
        #                outline=tuple(text_color[::-1]), width=1)
        
        # 序号和类别名称 - 使用ID替换序号
        class_name = item.get('classes', '未知')
        display_id = item.get('display_id', f'{index+1:04d}')
        status_text = "●" if is_recent else "○"  # 实心圆表示最近检测到
        
        # 处理长文本，确保适应格子宽度
        text = f"ID{display_id} {status_text} {class_name}"
        if len(text) > 18:  # 如果文本过长，截断类别名称
            class_name = class_name[:12] + "..."
            text = f"ID{display_id} {status_text} {class_name}"
        
        # 绘制文本，位置固定在格子内
        draw.text((cell_x + 5, cell_y + 5), text, font=item_font, fill=text_color_to_use)
    
    # 添加底部信息
    info_text = f"总计: {len(detection_list)} 个已确认目标"
    info_font = item_font
    draw.text((start_x + margin, h - margin - 30), info_text, font=info_font, fill=tuple(title_color[::-1]))
    
    # 转换回OpenCV格式
    new_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    return new_img
    
if __name__ == '__main__':
    pass
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # # basic params
    # parser.add_argument('--model_path', type=str, required= True, help='model path, could be .pt or .rknn file')
    # parser.add_argument('--target', type=str, default=None, help='target RKNPU platform')
    # parser.add_argument('--device_id', type=str, default=None, help='device id')
    
    # parser.add_argument('--img_show', action='store_true', default=False, help='draw the result and show')
    # parser.add_argument('--img_save', action='store_true', default=False, help='save the result')

    # # data params
    # parser.add_argument('--anno_json', type=str, default='../../../datasets/COCO/annotations/instances_val2017.json', help='coco annotation path')
    # # coco val folder: '../../../datasets/COCO//val2017'
    # parser.add_argument('--img_folder', type=str, default='../model', help='img folder path')
    # parser.add_argument('--coco_map_test', action='store_true', help='enable coco map test')

    # args = parser.parse_args()

    # # init model
    # model, platform = setup_model(args)

    # file_list = sorted(os.listdir(args.img_folder))
    # img_list = []
    # for path in file_list:
    #     if img_check(path):
    #         img_list.append(path)
    # co_helper = COCO_test_helper(enable_letter_box=True)

    # # run test
    # for i in range(len(img_list)):
    #     print('infer {}/{}'.format(i+1, len(img_list)), end='\r')

    #     img_name = img_list[i]
    #     img_path = os.path.join(args.img_folder, img_name)
    #     if not os.path.exists(img_path):
    #         print("{} is not found", img_name)
    #         continue

    #     img_src = cv2.imread(img_path)
    #     if img_src is None:
    #         continue

    #     '''
    #     # using for test input dumped by C.demo
    #     img_src = np.fromfile('./input_b/demo_c_input_hwc_rgb.txt', dtype=np.uint8).reshape(640,640,3)
    #     img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)
    #     '''

    #     # Due to rga init with (0,0,0), we using pad_color (0,0,0) instead of (114, 114, 114)
    #     pad_color = (0,0,0)
    #     img = co_helper.letter_box(im= img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #     # preprocee if not rknn model
    #     if platform in ['pytorch', 'onnx']:
    #         input_data = img.transpose((2,0,1))
    #         input_data = input_data.reshape(1,*input_data.shape).astype(np.float32)
    #         input_data = input_data/255.
    #     else:
    #         input_data = img

    #     outputs = model.rknn.inference([input_data])
    #     boxes, classes, scores = post_process(outputs)

    #     if args.img_show or args.img_save:
    #         print('\n\nIMG: {}'.format(img_name))
    #         img_p = img_src.copy()
    #         if boxes is not None:
    #             draw(img_p, co_helper.get_real_box(boxes), scores, classes)

    #         if args.img_save:
    #             if not os.path.exists('./result'):
    #                 os.mkdir('./result')
    #             result_path = os.path.join('./result', img_name)
    #             cv2.imwrite(result_path, img_p)
    #             print('Detection result save to {}'.format(result_path))
                        
    #         if args.img_show:
    #             cv2.imshow("full post process result", img_p)
    #             cv2.waitKeyEx(0)

    #     # record maps
    #     if args.coco_map_test is True:
    #         if boxes is not None:
    #             for i in range(boxes.shape[0]):
    #                 co_helper.add_single_record(image_id = int(img_name.split('.')[0]),
    #                                             category_id = coco_id_list[int(classes[i])],
    #                                             bbox = boxes[i],
    #                                             score = round(scores[i], 5).item()
    #                                             )

    # # calculate maps
    # if args.coco_map_test is True:
    #     pred_json = args.model_path.split('.')[-2]+ '_{}'.format(platform) +'.json'
    #     pred_json = pred_json.split('/')[-1]
    #     pred_json = os.path.join('./', pred_json)
    #     co_helper.export_to_json(pred_json)

    #     from py_utils.coco_utils import coco_eval_with_json
    #     coco_eval_with_json(args.anno_json, pred_json)

    # # release
    # model.release()
