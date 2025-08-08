import cv2
import time
import numpy as np
import cvui
from collections import defaultdict
from rknnlite.api import RKNNLite
from coco_utils import COCO_test_helper
from func import CLASSES
from PIL import Image, ImageDraw, ImageFont

# 配置参数
WINDOW_NAME = "RKNN Object Detection"
modelPath = "/home/cat/programs/rknn-multi-threaded/rknnModel/yolo11m_i8_train.rknn"
IMG_SIZE = 640
OBJ_THRESH = 0.25
NMS_THRESH = 0.45

class SimpleDetector:
    """简单的单线程检测器"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.rknn_lite = None
        self.co_helper = COCO_test_helper(enable_letter_box=True)
        self.init_model()
    
    def init_model(self):
        """初始化RKNN模型"""
        print("Loading RKNN model...")
        self.rknn_lite = RKNNLite()
        ret = self.rknn_lite.load_rknn(self.model_path)
        if ret != 0:
            print("Load RKNN model failed")
            exit(ret)
        
        ret = self.rknn_lite.init_runtime()
        if ret != 0:
            print("Init runtime environment failed")
            exit(ret)
        print("RKNN model loaded successfully")
    
    def detect_frame(self, frame):
        """检测单帧图像"""
        if self.rknn_lite is None:
            return frame, []
        
        try:
            # 预处理
            img_src = frame.copy()
            pad_color = (0, 0, 0)
            img_pre = self.co_helper.letter_box(im=img_src, new_shape=IMG_SIZE, pad_color=pad_color)
            img_pre = np.expand_dims(img_pre, axis=0)
            
            # 推理
            outputs = self.rknn_lite.inference(inputs=[img_pre])
            
            # 后处理
            result_frame, detections = self.process_detections(img_src, outputs)
            return result_frame, detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return frame, []
    
    def process_detections(self, img_src, outputs):
        """处理检测结果"""
        detections = []
        
        if outputs is None or len(outputs) == 0:
            return img_src, detections
        
        # 这里需要根据实际的模型输出格式来解析
        # 简化处理，假设已有process_image函数
        try:
            from rknn_image import process_image, CLASSES
            result_frame, detection_results = process_image(img_src, outputs, self.co_helper, overlay_mode=True, add_detection_list=False)
            
            # 提取检测结果
            if detection_results:
                for det in detection_results:
                    # 从rknn_image.py的process_image函数返回的格式是字典
                    # 包含"classes", "score", "last_seen", "display_id", "is_current_frame", "bbox"等键
                    class_name = det.get('classes', 'unknown')
                    score = det.get('score', 0.0)
                    bbox = det.get('bbox', [0, 0, 100, 100])  # [x1, y1, x2, y2]
                    
                    # 查找类别索引（需要将中文类别名映射回英文类别名）
                    class_id = -1
                    # 定义中文到英文的映射
                    class_map = {
                        "防护袋": "protective_bag",
                        "安全帽": "safety_helmet",
                        "刷子": "brush",
                        "手机": "mobile_phone",
                        "防护旗": "protective_flag",
                        "哨子": "whistle",
                        "扩音器": "loudhailer",
                        "钳子": "pliers",
                        "螺丝刀": "screwdriver",
                        "万用表": "multimeter",
                        "背包": "backpack",
                        "对讲机": "walkie-talkie",
                        "工具箱": "toolbox",
                        "探照灯": "spotlight",
                        "扳手": "wrench",
                        "医药箱": "medicine_box",
                        "手套": "glove",
                        "手电筒": "flashlight"
                    }
                    english_class_name = class_map.get(class_name, class_name)
                    
                    # 查找英文类别名在CLASSES中的索引
                    for i, cls in enumerate(CLASSES):
                        if cls.strip() == english_class_name.strip():
                            class_id = i
                            break
                    
                    if score > OBJ_THRESH and class_id >= 0:
                        detections.append({
                            'bbox': bbox,
                            'class_name': class_name,  # 保持中文显示
                            'score': float(score)
                        })
            
            return result_frame, detections
            
        except ImportError:
            # 如果没有process_image函数，使用简化的处理
            return self.simple_process(img_src, outputs)
    
    def simple_process(self, img_src, outputs):
        """简化的检测结果处理"""
        detections = []
        result_frame = img_src.copy()
        
        # 这里应该根据实际模型输出进行解析
        # 暂时返回空检测结果
        return result_frame, detections
    
    def release(self):
        """释放资源"""
        if self.rknn_lite:
            self.rknn_lite.release()

class DetectionApp:
    """检测应用主类"""
    
    def __init__(self):
        # 初始化摄像头
        # self.cap = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture('/home/cat/programs/rknn-multi-threaded/output_smooth.mp4')
        if not self.cap.isOpened():
            # 如果摄像头打不开，尝试使用视频文件
            self.cap = cv2.VideoCapture('./720p60hz.mp4')
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        
        # 初始化检测器
        self.detector = SimpleDetector(modelPath)
        
        # UI状态
        self.detecting = False
        self.detection_results = []
        self.current_frame = None
        self.detected_frame = None  # 存储检测结果帧
        self.show_detection_result = False  # 是否显示检测结果
        
        # 窗口尺寸设置
        self.frame_width = 640
        self.frame_height = 480
        
        # 初始化cvui
        cvui.init(WINDOW_NAME)
    
    def collect_frames_and_detect(self, duration=3.0, interval=0.1):
        """收集指定时间内的帧并进行检测"""
        frames_data = []
        start_time = time.time()
        
        print(f"开始收集{duration}秒内的帧...")
        
        while time.time() - start_time < duration:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 检测当前帧
            result_frame, detections = self.detector.detect_frame(frame)
            
            frames_data.append({
                'frame': result_frame,
                'detections': detections,
                'detection_count': len(detections),
                'timestamp': time.time()
            })
            
            # 等待指定间隔
            time.sleep(interval)
        
        print(f"收集完成，共{len(frames_data)}帧")
        
        # 找到检测结果最多的帧
        if frames_data:
            best_frame_data = max(frames_data, key=lambda x: x['detection_count'])
            return best_frame_data
        
        return None
    
    def draw_detection_results(self, frame, detections):
        """在帧上绘制检测结果"""
        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 尝试加载中文字体
        try:
            # 根据您的系统路径调整字体文件路径
            font = ImageFont.truetype("simhei.ttf", 20)
            label_font = ImageFont.truetype("simhei.ttf", 16)
        except:
            # 如果无法加载中文字体，则使用默认字体
            font = ImageFont.load_default()
            label_font = ImageFont.load_default()
        
        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            score = det['score']
            
            # 绘制边界框
            draw.rectangle([(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))], 
                          outline=(0, 255, 0), width=2)
            
            # 绘制标签
            label = f"{class_name} {score:.2f}"
            # 计算文本位置
            text_position = (int(bbox[0]), int(bbox[1]) - 20)
            # 绘制文本背景（可选）
            # draw.rectangle([text_position, (text_position[0] + 100, text_position[1] + 20)], 
            #               fill=(0, 0, 0))
            # 绘制文本
            draw.text(text_position, label, font=label_font, fill=(0, 255, 0))
        
        # 将PIL图像转换回OpenCV格式
        result_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        return result_frame
    
    def create_combined_image(self, original_frame, detection_frame=None):
        """创建拼接图像：左侧原始图像或绘制bbox的图像，右侧检测结果列表或空白"""
        # 调整左侧图像大小
        if detection_frame is not None and self.show_detection_result:
            # 如果有检测结果，左侧显示绘制了bbox的图像
            left_img = cv2.resize(detection_frame, (self.frame_width, self.frame_height))
        else:
            # 否则左侧显示原始图像
            left_img = cv2.resize(original_frame, (self.frame_width, self.frame_height))
        
        # 右侧图像处理
        if detection_frame is not None and self.show_detection_result:
            # 如果有检测结果，右侧显示检测目标列表
            right_img = self.create_detection_list_image()
        else:
            # 否则右侧显示空白背景
            right_img = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # 水平拼接两个图像
        combined_img = np.hstack([left_img, right_img])
        
        return combined_img
    
    def create_detection_list_image(self):
        """创建检测目标列表图像"""
        # 创建一个黑色背景的图像
        list_img = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(list_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 尝试加载中文字体
        try:
            # 根据您的系统路径调整字体文件路径
            title_font = ImageFont.truetype("simhei.ttf", 24)
            item_font = ImageFont.truetype("simhei.ttf", 20)
            small_font = ImageFont.truetype("simhei.ttf", 16)
        except:
            # 如果无法加载中文字体，则使用默认字体
            title_font = ImageFont.load_default()
            item_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # 如果有检测结果，绘制检测目标列表
        if self.detection_results:
            # 添加标题
            title = "检测目标列表"
            title_with_count = f"{title} 核准{len(self.detection_results)}个"
            draw.text((20, 30), title_with_count, font=title_font, fill=(255, 255, 255))
            
            # 绘制分隔线
            draw.line([(20, 50), (self.frame_width - 20, 50)], fill=(200, 200, 200), width=1)
            
            # 绘制每个检测目标
            start_y = 80
            for i, det in enumerate(self.detection_results):
                if i >= 10:  # 最多显示10个目标
                    break
                    
                # 计算位置
                row = i // 2
                col = i % 2
                x = 20 + col * (self.frame_width // 2 - 30)
                y = start_y + row * 35
                
                # 绘制目标信息
                class_name = det['class_name']
                score = det['score']
                text = f"ID{i+1:04d} {class_name} {score:.2f}"
                
                # 绘制文本
                draw.text((x, y), text, font=item_font, fill=(200, 200, 200))
            
            # 添加总计信息
            total_text = f"总计: {len(self.detection_results)} 个已确认目标"
            draw.text((20, self.frame_height - 30), total_text, font=small_font, fill=(255, 255, 255))
        else:
            # 没有检测结果时显示提示文本
            draw.text((50, self.frame_height // 2 - 10), "暂无确认目标", font=title_font, fill=(255, 255, 255))
        
        # 将PIL图像转换回OpenCV格式
        list_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        return list_img
    
    def run(self):
        """主运行循环"""
        print("Detection app started. Press 'q' to quit.")
        
        while True:
            # 获取当前帧
            ret, frame = self.cap.read()
            if not ret:
                print("无法获取摄像头画面")
                break
            
            self.current_frame = frame.copy()
            
            # 创建拼接图像
            if self.show_detection_result and self.detected_frame is not None:
                combined_img = self.create_combined_image(frame, self.detected_frame)
            else:
                combined_img = self.create_combined_image(frame)
            
            # 创建UI画布
            canvas_height = combined_img.shape[0] + 150  # 为按钮留出空间
            canvas_width = combined_img.shape[1] + 100
            canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 50
            
            # 显示拼接图像
            y_offset = 50
            x_offset = 50
            canvas[y_offset:y_offset+combined_img.shape[0], x_offset:x_offset+combined_img.shape[1]] = combined_img
            
            # 按钮布局
            button_y = y_offset + combined_img.shape[0] + 20
            if not self.show_detection_result:
                # 实时模式：显示检测按钮
                button_pressed = cvui.button(canvas, x_offset, button_y, 150, 40, "Start Detection")
                return_pressed = False
            else:
                # 检测结果模式：显示返回按钮
                button_pressed = False
                return_pressed = cvui.button(canvas, x_offset, button_y, 150, 40, "Return to Live")
            
            # 状态显示
            status_x = x_offset + 180
            if self.detecting:
                cvui.text(canvas, status_x, button_y + 15, "Detecting...", 0.5, 0x00ff00)
            elif self.show_detection_result:
                cvui.text(canvas, status_x, button_y + 15, "Detection Results", 0.5, 0x00ffff)
            else:
                cvui.text(canvas, status_x, button_y + 15, "Live Camera", 0.5, 0xffffff)
            
            # 处理按钮点击
            if button_pressed and not self.detecting:
                self.detecting = True
                print("开始检测...")
                
                # 执行检测
                best_frame_data = self.collect_frames_and_detect()
                
                if best_frame_data:
                    self.detected_frame = best_frame_data['frame']
                    self.detection_results = best_frame_data['detections']
                    self.show_detection_result = True  # 切换到检测结果模式
                    print(f"检测完成，找到{len(self.detection_results)}个目标")
                else:
                    print("检测失败")
                    self.detection_results = []
                    self.show_detection_result = False
                
                self.detecting = False
            
            # 处理返回按钮
            if return_pressed:
                self.show_detection_result = False
                self.detection_results = []
                self.detected_frame = None
                print("返回实时模式")
            
            # 更新UI
            cvui.update()
            cv2.imshow(WINDOW_NAME, canvas)
            
            # 检查退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
        
        # 清理资源
        self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.detector.release()
        print("资源已释放")

def main():
    """主函数"""
    try:
        app = DetectionApp()
        app.run()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序异常: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()