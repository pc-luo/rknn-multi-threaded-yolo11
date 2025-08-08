#以下代码改自https://github.com/rockchip-linux/rknn-toolkit2/tree/master/examples/onnx/yolov5
import time

import cv2
import numpy as np
from rknn_image import process_image, CLASSES_CHINESE
from PIL import Image, ImageDraw, ImageFont
OBJ_THRESH, NMS_THRESH, IMG_SIZE = 0.25, 0.45, 640

CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")


# 
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))


def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = input[..., 4]
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = input[..., 5:]

    box_xy = input[..., :2] *2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)

    box_wh = pow(input[..., 2:4] *2, 2)
    box_wh = box_wh * anchors

    return np.concatenate((box_xy, box_wh), axis=-1), box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    return boxes[_class_pos], classes[_class_pos], (class_max_score * box_confidences)[_class_pos]


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

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
    return np.array(keep)


def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)


def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        # print('class: {}, score: {}'.format(CLASSES[cl], score))
        # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)

        cv2.rectangle(image, (top, left), (int(right), int(bottom)), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border
    return im
    # return im, ratio, (dw, dh)

def myFunc(rknn_lite, IMG, co_helper):
    # IMG = cv2.cvtColor(IMG, cv2.COLOR_BGR2RGB)
    # # 等比例缩放
    # # IMG = letterbox(IMG)
    # # 强制放缩
    # IMG = cv2.resize(IMG, (IMG_SIZE, IMG_SIZE))

    # input_data = IMG.transpose((2, 0, 1))
    # input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
    # input_data = input_data / 255.
    img_src = IMG
    frame = img_src  # 初始化frame变量，避免未定义错误
    
    pad_color = (0,0,0)
    img_pre = co_helper.letter_box(im=img_src.copy(), new_shape=IMG_SIZE, pad_color=pad_color)
    img_pre = np.expand_dims(img_pre, axis=0)
    try:
        start_time = time.time()
        outputs = rknn_lite.inference(inputs=[img_pre])
        # print("inference time: ", time.time() - start_time)
        frame, detection_results = process_image(img_src, outputs, co_helper, overlay_mode=True, add_detection_list=False)
        
        # 创建拼接图像（左侧为处理后的图像，右侧为检测列表）
        # 调整图像大小
        h, w = img_src.shape[:2]
        # left_img = cv2.resize(frame, (w//2, h))
        left_img = frame
        # 创建右侧检测列表图像
        right_img = create_detection_list_image(detection_results, w//2, h)
        
        # 水平拼接两个图像
        frame = np.hstack([left_img, right_img])
    except Exception as e:
        print("error: ", e)
        frame = img_src  # 发生异常时返回原始图像
    # if frame is not None:
    #     end_time = time.time()
    #     elapsed_time = end_time - self.start_time
    #     if elapsed_time > 1:  # 每秒更新一次帧率
    #         fps = frame_count / elapsed_time
    #         # 重置计数器和时间戳
    #         self.frame_count = 0
    #         self.start_time = time.time()
    #     fps_text = f"FPS: {int(self.fps)}"
    #     cv2.putText(self.frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # if outputs is None:
    #     return IMG

    # input0_data = outputs[0].reshape([3, -1]+list(outputs[0].shape[-2:]))
    # input1_data = outputs[1].reshape([3, -1]+list(outputs[1].shape[-2:]))
    # input2_data = outputs[2].reshape([3, -1]+list(outputs[2].shape[-2:]))

    # input_data = list()
    # input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    # input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    # input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

    # boxes, classes, scores = yolov5_post_process(input_data)

    # IMG = cv2.cvtColor(IMG, cv2.COLOR_RGB2BGR)
    return frame

def create_detection_list_image(detection_results, width, height):
    """创建检测目标列表图像"""
    # 创建一个黑色背景的图像
    list_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 转换为PIL图像以便绘制中文
    img_pil = Image.fromarray(cv2.cvtColor(list_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 尝试加载中文字体，如果失败则使用默认字体
    try:
        # 这里使用一个常见的中文字体路径，你可能需要根据你的系统调整
        font_path = "/home/cat/programs/rknn-multi-threaded/simhei.ttf"
        title_font = ImageFont.truetype(font_path, 24)
        item_font = ImageFont.truetype(font_path, 20)
        large_font = ImageFont.truetype(font_path, 28)
    except:
        # 如果无法加载中文字体，则使用默认字体
        title_font = ImageFont.load_default()
        item_font = ImageFont.load_default()
        large_font = ImageFont.load_default()
    
    # 如果有检测结果，绘制检测目标列表
    if detection_results:
        # 添加标题
        title = "检测目标列表"
        title_with_count = f"{title} 核准{len(detection_results)}个"
        draw.text((10, 20), title_with_count, font=title_font, fill=(255, 255, 255))
        
        # 绘制分隔线
        draw.line([(10, 50), (width - 10, 50)], fill=(200, 200, 200), width=1)
        
        # 绘制每个检测目标
        start_y = 70
        for i, det in enumerate(detection_results):
            if i >= 10:  # 最多显示10个目标
                break
                
            # 计算位置
            row = i // 1
            col = i % 1
            x = 10 + col * (width // 2 - 20)
            y = start_y + row * 30
            
            # 绘制目标信息
            class_name = det.get('classes', '未知')
            score = det.get('score', 0)
            display_id = det.get('display_id', f'{i+1:04d}')
            text = f"ID{display_id} {class_name} {score:.2f}"
            
            # 绘制文本
            draw.text((x, y), text, font=item_font, fill=(200, 200, 200))
        
        # 添加总计信息
        total_text = f"总计: {len(detection_results)} 个已确认目标"
        draw.text((10, height - 30), total_text, font=item_font, fill=(255, 255, 255))
    else:
        # 没有检测结果时显示提示文本
        draw.text((20, height // 2 - 15), "暂无确认目标", font=large_font, fill=(255, 255, 255))
    
    # 转换回OpenCV格式
    list_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    return list_img
