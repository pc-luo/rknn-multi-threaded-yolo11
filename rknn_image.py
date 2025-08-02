import urllib
import time
import sys
import numpy as np
import cv2
from rknnlite.api import RKNNLite
import os
import cv2
import argparse
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

from collections import defaultdict
detection_history = defaultdict(list)  # 格式: {"目标名": [帧1结果, 帧2结果...]}
FRAME_WINDOW_SIZE = 5  # 滑动窗口大小（帧数）
MIN_DETECT_COUNT = 3   # 最小确认次数


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
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        # print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
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
def process_image(image, outputs, coco_helper):
    # post process
    # img_src = image.copy()
    # input0_data = outputs[0]
    # input1_data = outputs[1]
    # input2_data = outputs[2]

    # input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:]))
    # input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
    # input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))

    # input_data = list()
    # input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    # input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    # input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

    boxes, classes, scores = post_process(outputs)

    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # if boxes is not None:
    #     draw(image, boxes, scores, classes)
    detection_results = []
    if boxes is not None:
        draw(image, coco_helper.get_real_box(boxes), scores, classes)
        for cl in classes:
            detection_results.append({
                'classes': CLASSES_CHINESE[CLASSES[cl]]}
                )
        image = add_detection_list_to_image(image=image, detection_list=detection_results)
        return image, detection_results
    else:
        image = add_detection_list_to_image(image=image, detection_list=detection_results)
        return image, None
        
def add_detection_list_to_image(image, detection_list, add_width=480,
                               background_color=(0, 0, 0),  # 深灰色背景
                               text_color=(255, 255, 255),     # 白色文字
                               font_scale=1,                 # 字体大小
                               thickness=1):                  # 文字粗细
    """
    在图像右侧添加纯色背景和检测对象列表
    
    参数:
        image_path: 输入图像路径
        detection_list: 检测对象列表，例如 ['person', 'car', 'dog']
        output_path: 输出图像路径
        background_color: 背景颜色 (B, G, R)
        text_color: 文字颜色 (B, G, R)
        font_scale: 字体大小
        thickness: 文字粗细
    """
    import time   # 放在文件顶部
    # 读取原始图像
    h, w = image.shape[:2]

    # 计算新图像的宽度（增加100像素）
    new_w = w + add_width
    new_h = h
    
    # 创建新图像（右侧100像素为纯色背景）
    new_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    new_img[:, :w] = image  # 左侧放置原图
    new_img[:, w:] = background_color  # 右侧填充背景色

    # 设置字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 计算文字高度
    (text_width, text_height), _ = cv2.getTextSize("A", font, font_scale, thickness)
    # text_height += 5
    # if detection_list:
    # # 在右侧区域垂直居中显示检测列表
    #     start_y = (new_h - len(detection_list) * text_height) // 2
        
    #     for index, text in enumerate(detection_list):
    #         y = start_y + index * text_height
    #         cv2.putText(new_img, f"{index}.{text['classes']}", 
    #                 (w + 10, y + text_height),  # x坐标留10像素边距
    #                 font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    # (_, text_height), _ = cv2.getTextSize("测", font_face, font_scale, thickness)
    line_height = text_height + 5
    
    if detection_list:
        total_height = len(detection_list) * line_height
        start_y = max((new_h - total_height) // 2, 0)

        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        # 尝试加载中文字体
        font = ImageFont.truetype("/home/cat/programs/rknn-multi-threaded/simhei.ttf", int(text_height))
        for index, item in enumerate(detection_list):
            y = start_y + index * line_height
            draw.text((w + 10, y), f"{index+1}. {item['classes']}", 
                        font=font, fill=tuple(text_color[::-1]))
        # file_name = f"pil{int(time.time()*1000)}.jpg"

        # # JPEG 压缩保存：质量 0–100，数值越小压缩率越高
        # img_pil.save(f"test/{file_name}", format="JPEG", quality=100)
        # 转换回OpenCV格式
        new_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 在保存前一行替换原来的 file_name
        # file_name = f"{int(time.time()*1000)}.jpg"
        # cv2.imwrite(f"test/cv{file_name}", new_img)
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
