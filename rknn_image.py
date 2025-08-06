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
        """生成目标ID"""
        return f"{class_name}_{int(box[0])}_{int(box[1])}_{int(time.time() % 10000)}"
    
    def update(self, detections, current_time):
        """更新跟踪器状态
        
        Args:
            detections: 当前帧的检测结果 [(box, class_name, score), ...]
            current_time: 当前时间戳
        """
        # 更新的目标ID列表
        updated_target_ids = set()
        
        # 1. 处理当前帧的检测
        for box, class_name, score in detections:
            # 查找最佳匹配
            matched_id = self.find_best_match(box, class_name, current_time)
            
            if matched_id:
                # 更新现有目标
                self.tracked_targets[matched_id].update({
                    'bbox': box,
                    'last_seen': current_time,
                    'score': score,
                    'detection_count': self.tracked_targets[matched_id].get('detection_count', 0) + 1
                })
                updated_target_ids.add(matched_id)
            else:
                # 创建新目标
                target_id = self.generate_target_id(class_name, box)
                self.tracked_targets[target_id] = {
                    'class_name': class_name,
                    'bbox': box,
                    'score': score,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'confirmed': False,
                    'detection_count': 1,
                    'confirmed_at': None
                }
                updated_target_ids.add(target_id)
        
        # 2. 更新检测历史和确认状态
        for target_id in updated_target_ids:
            target = self.tracked_targets[target_id]
            
            # 更新检测历史
            if target_id not in self.detection_history:
                self.detection_history[target_id] = deque(maxlen=self.confirmation_window)
            self.detection_history[target_id].append(current_time)
            
            # 检查确认条件
            if not target['confirmed']:
                recent_detections = len([t for t in self.detection_history[target_id] 
                                       if current_time - t <= self.confirmation_window])
                if recent_detections >= self.min_confirmations:
                    target['confirmed'] = True
                    target['confirmed_at'] = current_time
        
        # 3. 清理超时目标
        self.cleanup_timeout_targets(current_time)
        
        # 4. 返回持久化显示的目标列表
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
            del self.tracked_targets[target_id]
            if target_id in self.detection_history:
                del self.detection_history[target_id]
    
    def get_persistent_targets(self):
        """获取需要持久化显示的目标列表"""
        persistent_targets = []
        
        for target_id, target_info in self.tracked_targets.items():
            if target_info['confirmed']:
                persistent_targets.append({
                    'id': target_id,
                    'classes': target_info['class_name'],
                    'bbox': target_info['bbox'],
                    'score': target_info['score'],
                    'last_seen': target_info['last_seen']
                })
        
        return persistent_targets
    
    def get_all_targets(self):
        """获取所有目标（包括已确认和未确认的）"""
        return dict(self.tracked_targets)


# 全局跟踪器实例
target_tracker = TargetTracker(confirmation_window=10, min_confirmations=3, persistence_timeout=10.0) 

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

def process_image(image, outputs, coco_helper):
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
            class_name = CLASSES_CHINESE[CLASSES[cl]]
            detections.append((box, class_name, score))

    # 2. 更新目标跟踪器
    persistent_targets = target_tracker.update(detections, current_time)
    
    # 3. 获取所有目标（包括未确认的）用于调试和显示
    all_targets = target_tracker.get_all_targets()

    # 4. 绘制目标
    # 4.1 绘制已确认的目标（蓝色边界框）
    for target in persistent_targets:
        box = target['bbox']
        class_name = target['classes']
        score = target['score']
        
        # 绘制蓝色边界框表示已确认的目标
        cv2.rectangle(image, 
                     (int(box[0]), int(box[1])), 
                     (int(box[2]), int(box[3])), 
                     (255, 0, 0), 2)  # 蓝色
        # 绘制标签
        cv2.putText(image, f"{class_name} {score:.2f}", 
                    (int(box[0]), int(box[1]) - 6), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
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
            # 绘制标签
            cv2.putText(image, f"{class_name}?", 
                        (int(box[0]), int(box[1]) - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 5. 生成右侧标签列表（仅显示已确认的目标）
    confirmed_list = [{"classes": target['classes'], "score": target['score']} 
                     for target in persistent_targets]
    image = add_detection_list_to_image(image=image, detection_list=confirmed_list)

    return image, confirmed_list if confirmed_list else None
        
def add_detection_list_to_image(image, detection_list, add_width=480,
                               background_color=(30, 30, 30),   # 深灰色背景
                               title_color=(255, 255, 255),    # 白色标题
                               text_color=(200, 200, 200),     # 浅灰色文字
                               highlight_color=(255, 255, 0),  # 黄色高亮
                               title="检测目标列表"):
    """
    在图像右侧添加纯色背景和持久化检测对象列表
    
    参数:
        image: 输入图像
        detection_list: 检测对象列表，例如 [{'classes': 'person', 'score': 0.85}, ...]
        add_width: 右侧扩展宽度
        background_color: 背景颜色 (B, G, R)
        title_color: 标题颜色 (B, G, R)
        text_color: 文字颜色 (B, G, R)
        highlight_color: 高亮颜色 (B, G, R)
        title: 标题文本
    """
    import time
    h, w = image.shape[:2]
    
    # 创建新图像
    new_w = w + add_width
    new_h = h
    new_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    new_img[:, :w] = image
    new_img[:, w:] = background_color
    
    # 设置字体
    cv2_font = cv2.FONT_HERSHEY_SIMPLEX
    
    if not detection_list:
        # 显示"暂无确认目标"
        img_pil = Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("/home/cat/programs/rknn-multi-threaded/simhei.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        text = "暂无确认目标"
        draw.text((w + 20, h//2 - 20), text, font=font, fill=tuple(title_color[::-1]))
        new_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return new_img
    
    # 计算布局
    title_height = 60
    item_height = 40
    margin = 20
    padding = 15
    
    # 绘制标题
    img_pil = Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        title_font = ImageFont.truetype("/home/cat/programs/rknn-multi-threaded/simhei.ttf", 32)
        item_font = ImageFont.truetype("/home/cat/programs/rknn-multi-threaded/simhei.ttf", 24)
    except:
        title_font = ImageFont.load_default()
        item_font = ImageFont.load_default()
    
    # 绘制标题
    draw.text((w + margin, padding), title, font=title_font, fill=tuple(title_color[::-1]))
    
    # 绘制分隔线
    draw.line([(w + margin, padding + title_height - 10), 
               (w + add_width - margin, padding + title_height - 10)], 
              fill=tuple(text_color[::-1]), width=2)
    
    # 显示目标列表
    current_y = padding + title_height
    
    for index, item in enumerate(detection_list):
        # 计算时间差
        time_since_last_seen = time.time() - item.get('last_seen', time.time())
        is_recent = time_since_last_seen < 2.0  # 最近2秒检测到的目标高亮显示
        
        # 选择颜色
        text_color_to_use = tuple(highlight_color[::-1]) if is_recent else tuple(text_color[::-1])
        
        # 序号和类别名称
        class_name = item.get('classes', '未知')
        score = item.get('score', 0.0)
        status_text = "●" if is_recent else "○"  # 实心圆表示最近检测到
        
        text = f"{index+1:2d}. {status_text} {class_name}"
        if score > 0:
            text += f" ({score:.2f})"
        
        # 绘制文本
        draw.text((w + margin + 10, current_y), text, font=item_font, fill=text_color_to_use)
        
        current_y += item_height
    
    # 添加底部信息
    info_text = f"总计: {len(detection_list)} 个已确认目标"
    info_font = item_font
    draw.text((w + margin, new_h - margin - 30), info_text, font=info_font, fill=tuple(title_color[::-1]))
    
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
