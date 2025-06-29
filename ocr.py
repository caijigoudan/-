import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageFilter
from paddleocr import PaddleOCR
from fpdf import FPDF
import time
import threading
import socket
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
import pymysql

# 连接数据库
connection = pymysql.connect(
    host='localhost',  # 数据库主机地址
    user='root',  # 数据库用户名
    password='gzy20020228',  # 数据库密码
    database='ocr',  # 数据库名称
    charset='utf8mb4',  # 字符编码
    cursorclass=pymysql.cursors.DictCursor  # 指定游标类型，返回字典格式的数据
)


# 初始化大语言模型
# 参数说明:
# zhipuai_api_key - 智谱AI的API密钥
# model - 使用的模型名称(glm-4-flash是轻量级模型)
# temperature - 控制生成文本的随机性(0.0-1.0)
# top_p - 控制生成文本的多样性(0.0-1.0)
llm = ChatZhipuAI(
    zhipuai_api_key='c2307906dbf8486291c9b9ed6c46bc98.xcbSzlxvkmKMu7Gk',
    model="glm-4-flash",
    temperature=0.8,  # 较高的随机性使输出更有创造性
    top_p=0.9,  # 较高的多样性使输出更丰富
)

# 构建翻译提示模板
# 该模板定义了大模型的任务和行为
# {text}是占位符，将被实际文本替换
template = """
你是一个专业的翻译助手，你的任务是将输入的文本翻译成中文。
请将以下文本翻译成中文：
{text},并返回文本的重点"""

# 从模板创建提示
prompt = ChatPromptTemplate.from_template(template)

# 搭建处理链: 提示 -> 大模型
# 该链将按顺序执行模板填充和大模型调用
chain = prompt | llm


class OCRApp:
    def __init__(self, master):
        self.master = master
        master.title("OCR专业版 v2.2")  # 设置窗口标题
        master.geometry("1000x800")  # 设置窗口大小
        master.protocol("WM_DELETE_WINDOW", self.on_close)  # 设置关闭窗口回调

        # 初始化变量
        self.photo_path = ""  # 原始图片路径
        self.sharpened_path = ""  # 增强后图片路径
        self.ocr_result = ""  # OCR识别结果
        self.base_dir = 'wait_use'  # 默认存储目录
        self.create_storage_dir()  # 创建存储目录

        # OCR引擎状态
        self.ocr_engine = None  # PaddleOCR实例
        self.ocr_ready = False  # OCR引擎是否就绪
        self.models_loaded = 0  # 已加载模型数量
        self.init_success = False  # 初始化是否成功

        # 界面初始化（必须先创建控件）
        self.progress = None  # 进度条控件
        self.create_widgets()  # 创建所有界面组件

        # 网络检测
        if self.check_network():  # 检查网络连接
            self.init_ocr_engine()  # 初始化OCR引擎
        else:
            master.after(100, self._shutdown)  # 无网络则关闭程序

        # 摄像头资源
        self.cap = None  # OpenCV视频捕获对象
        self.is_capturing = False  # 是否正在捕获视频
        self.current_frame = None  # 当前视频帧

    def check_network(self):
        """网络连接检测"""
        try:
            socket.create_connection(("www.baidu.com", 80), timeout=3)
            return True
        except OSError:
            messagebox.showwarning("网络中断",
                                   "需要互联网连接以下载OCR引擎\n请检查网络后重新启动程序")
            return False

    def create_storage_dir(self):
        """创建存储目录"""
        try:
            os.makedirs(self.base_dir, exist_ok=True)# 确保目录存在
            if not os.access(self.base_dir, os.W_OK):# 检查目录是否允许写入
                raise PermissionError("目录写入权限被拒绝")
        except Exception as e:
            self.base_dir = os.path.join(os.path.expanduser("~"), "Desktop")
            os.makedirs(self.base_dir, exist_ok=True)
            messagebox.showinfo("路径变更", f"存储路径已切换至桌面：{self.base_dir}")

    def init_ocr_engine(self):
        """改进的OCR引擎初始化"""
        # 更新GUI状态
        def update_gui_status(msg, color="black"):
            # 使用after方法确保GUI线程安全更新
            # 参数0表示立即执行，lambda创建匿名函数
            self.master.after(0, lambda: self.status_bar.config(
                # 格式化状态文本: [HH:MM] + 消息内容
                # time.strftime格式化当前时间为小时:分钟格式
                text=f"[{time.strftime('%H:%M')}] {msg}",
                # 设置状态文本颜色
                foreground=color
            ))

        def load_models():
            try:
                # 组件有效性验证 - 检查进度条组件是否存在且有效
                if self.progress is None or not self.progress.winfo_exists():
                    raise AttributeError("进度条组件异常")

                # 更新GUI状态显示初始化中
                update_gui_status("正在初始化引擎...")
                # 启动进度条动画，参数10控制动画速度
                self.progress.start(10)

                # 自动下载并初始化PaddleOCR模型
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True, # 启用角度分类器
                    lang='ch', # 使用中文模型
                    use_gpu=False, # 禁用GPU加速
                    enable_mkldnn=True, # 启用Intel MKL-DNN加速
                    show_log=True # 显示日志信息
                )

                # 模拟模型加载过程，提供用户反馈
                # 依次加载检测、识别和分类三个模型
                for stage in ["检测模型", "识别模型", "分类模型"]:
                    time.sleep(1)  # 模拟加载延迟
                    self.models_loaded += 1  # 更新已加载模型计数
                    update_gui_status(f"正在加载{stage} ({self.models_loaded}/3)")

                # 预加载测试 - 使用空白图像测试OCR引擎
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                self.ocr_engine.ocr(test_img)

                # 标记OCR引擎为就绪状态
                self.ocr_ready = True
                update_gui_status("OCR引擎就绪", "green")
                # 停止进度条动画
                self.progress.stop()

            except Exception as e:
                # 异常处理：停止进度条并显示错误信息
                self.progress.stop() if self.progress else None
                # 构造详细的错误消息
                error_msg = f"""初始化失败！
                可能原因：
                1. 缺少VC++运行库（需安装2015+版本）
                2. 磁盘空间不足（需300MB以上）
                3. 防火墙阻止下载

                错误详情：{str(e)}"""
                # 在GUI线程显示错误对话框
                self.master.after(0, lambda: messagebox.showerror(
                    "初始化失败", error_msg))
                # 更新状态栏显示失败信息
                update_gui_status("初始化失败", "red")
                # 关闭应用程序
                self._shutdown()

        if self.progress:  # 确认组件已存在
            threading.Thread(target=load_models, daemon=True).start()
        else:
            self._shutdown()

    def create_widgets(self):
        """优化后的用户界面布局"""
        # 创建主框架容器
        main_frame = ttk.Frame(self.master)
        # pack布局管理器 fill=tk.BOTH表示填充X和Y方向，expand=True允许框架随窗口缩放 padx/pady=10设置内边距为10像素
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建左侧控制面板区域，使用LabelFrame添加标题边框，width=240设置固定宽度240像素
        control_frame = ttk.LabelFrame(main_frame, text="操作面板", width=240)
        # side=tk.LEFT表示靠左对齐，fill=tk.Y表示垂直方向填充
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # 创建右侧显示区域框架
        display_frame = ttk.Frame(main_frame)
        # side=tk.RIGHT表示靠右对齐
        # fill=tk.BOTH+expand=True允许框架随窗口缩放
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 创建图像预览，背景色设置为深灰色(#333)
        self.preview_label = ttk.Label(display_frame, background='#333')
        # fill=tk.BOTH+expand=True使标签填满整个显示区域
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # 创建识别结果区域，使用LabelFrame添加标题边框
        # 内部包含一个Text控件用于显示OCR结果
        result_frame = ttk.LabelFrame(display_frame, text="识别结果")
        result_frame.pack(fill=tk.BOTH, expand=True)
        # wrap=tk.WORD实现自动换行，font设置字体为微软雅黑10号
        self.result_text = tk.Text(result_frame, wrap=tk.WORD, font=('微软雅黑', 10))
        # padx/pady=5设置内边距为5像素
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 定义按钮统一样式：水平填充(fill=tk.X)，左右边距5像素，上下边距3像素
        btn_style = {'fill': tk.X, 'padx': 5, 'pady': 3}
        
        # 创建功能按钮，每个按钮绑定对应的方法
        # command参数指定按钮点击时调用的方法
        ttk.Button(control_frame, text="启动摄像头", command=self.toggle_camera).pack(**btn_style)
        ttk.Button(control_frame, text="拍摄照片", command=self.capture_image).pack(**btn_style)
        ttk.Button(control_frame, text="选择图片", command=self.select_image).pack(**btn_style)
        ttk.Button(control_frame, text="图像增强", command=self.enhance_image).pack(**btn_style)
        ttk.Button(control_frame, text="文字识别", command=self.run_ocr).pack(**btn_style)
        ttk.Button(control_frame, text="生成报告", command=self.generate_report).pack(**btn_style)
        ttk.Button(control_frame, text="翻译文本", command=self.translate_text).pack(**btn_style)

        # 创建水平进度条，用于显示模型加载进度
        # mode='indeterminate'表示不确定进度模式
        # orient=tk.HORIZONTAL设置水平方向，length=200设置长度为200像素
        self.progress = ttk.Progressbar(
            control_frame,
            mode='indeterminate',
            orient=tk.HORIZONTAL,
            length=200
        )
        # fill=tk.X表示水平填充，pady=5设置上下边距为5像素
        self.progress.pack(fill=tk.X, pady=5)

        # 创建状态栏标签，显示程序状态信息
        # foreground="green"设置初始文本颜色为绿色
        # anchor=tk.W设置文本左对齐
        # side=tk.BOTTOM+fill=tk.X使标签位于底部并水平填充
        self.status_bar = ttk.Label(
            control_frame,
            text="就绪",
            foreground="green",
            anchor=tk.W
        )
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def toggle_camera(self):
        """摄像头开关控制"""
        if not self.is_capturing:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        try:
            # 初始化摄像头设备，参数0表示使用默认摄像头
            self.cap = cv2.VideoCapture(0)
            # 检查摄像头是否成功打开
            if not self.cap.isOpened():
                # 如果摄像头打开失败，抛出IOError异常
                raise IOError("无法访问摄像头")
            # 设置摄像头捕获状态为True，表示正在捕获视频
            self.is_capturing = True
            # 创建并启动一个后台线程来持续更新预览画面，target参数指定线程要执行的函数
            self.capture_thread = threading.Thread(target=self.update_preview)
            # 启动线程，开始捕获视频帧
            self.capture_thread.start()
            # 更新状态栏显示摄像头已开启，并设置文本颜色为蓝色
            self.status_bar.config(text="摄像头已开启", foreground="blue")
        except Exception as e:
            messagebox.showerror("摄像头错误", str(e))

    def stop_camera(self):
        self.is_capturing = False
        if self.cap:
            self.cap.release()
        self.status_bar.config(text="摄像头已关闭", foreground="gray")

    def update_preview(self):
        """持续更新摄像头预览画面的主循环"""
        while self.is_capturing:  # 当摄像头捕获标志为True时持续循环
            # 从摄像头读取一帧图像，ret表示读取是否成功，frame是捕获的图像数据
            ret, frame = self.cap.read()
            if ret:  # 如果成功读取到帧
                # 保存当前帧到实例变量，供其他方法使用
                self.current_frame = frame
                # 将BGR色彩空间转换为RGB，因为OpenCV使用BGR而PIL使用RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 将numpy数组转换为PIL图像对象
                img = Image.fromarray(frame)
                # 缩放图像到最大800x600像素，保持宽高比
                img.thumbnail((800, 600))
                # 将PIL图像转换为Tkinter兼容的PhotoImage对象
                photo = ImageTk.PhotoImage(img)
                # 更新预览标签的图像显示
                self.preview_label.configure(image=photo)
                # 保持对photo的引用，防止被垃圾回收
                self.preview_label.image = photo

    def capture_image(self):
        # 检查当前帧是否为空（摄像头未开启）
        if self.current_frame is None:
            messagebox.showwarning("警告", "请先开启摄像头")
            return
            
        try:
            # 生成带时间戳的图片文件名
            filename = f"capture_{time.strftime('%Y%m%d%H%M%S')}.jpg"
            # 拼接完整图片保存路径
            self.photo_path = os.path.join(self.base_dir, filename)
            # 使用OpenCV保存当前帧为图片
            cv2.imwrite(self.photo_path, self.current_frame)
            
            # 准备数据库插入数据
            file_size = os.path.getsize(self.photo_path)  # 获取文件大小（字节）
            file_type = os.path.splitext(filename)[1][1:]  # 提取文件扩展名（不含点）
            create_time = time.strftime('%Y-%m-%d %H:%M:%S')  # 当前格式化时间
            
            # 数据库操作
            with connection.cursor() as cursor:
                # 查询当前记录数作为新文件ID的基础
                cursor.execute("SELECT COUNT(*) AS count FROM filemetadata")
                result = cursor.fetchone()
                file_id = result['count'] + 1  # 新文件ID = 当前记录数+1
                
                # 准备SQL插入语句
                sql = """
                INSERT INTO filemetadata (file_id, file_path, file_type, file_size, create_time)
                VALUES (%s, %s, %s, %s, %s)
                """
                # 执行SQL插入
                cursor.execute(sql, (file_id, self.photo_path, file_type, file_size, create_time))
                connection.commit()  # 提交事务
            
            # 更新界面状态
            self.status_bar.config(text=f"已保存: {filename}", foreground="green")
            # 停止摄像头捕获
            self.stop_camera()
            # 显示保存的图片
            self.show_image(self.photo_path)
        except Exception as e:
            # 捕获并显示异常信息
            messagebox.showerror("保存失败", str(e))

    def select_image(self):
        # 使用文件对话框选择图片文件
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            # 保存文件路径到实例变量
            self.photo_path = file_path
            
            # 插入数据库记录
            file_size = os.path.getsize(file_path)
            file_type = os.path.splitext(file_path)[1][1:]
            create_time = time.strftime('%Y-%m-%d %H:%M:%S')
        
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) AS count FROM filemetadata")
                result = cursor.fetchone()
                file_id = result['count'] + 1
                
                sql = """
                INSERT INTO filemetadata (file_id, file_path, file_type, file_size, create_time)
                VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (file_id, file_path, file_type, file_size, create_time))
                connection.commit()
            
            # 在界面显示选中的图片
            self.show_image(file_path)
            # 更新状态栏显示已选择图片
            self.status_bar.config(text="已选择图片", foreground="blue")

    def enhance_image(self):
        if not self.photo_path:
            messagebox.showwarning("错误", "请先选择图片")
            return
        try:
            # 加载并显示原始图片
            # 使用OpenCV读取图片文件，返回numpy数组格式的图片数据
            original_img = cv2.imread(self.photo_path)
            if original_img is None:
                raise ValueError("无法读取图片文件")
            # 在界面中显示原始图片
            self.show_image(self.photo_path)
            self.status_bar.config(text="原始图片已加载", foreground="blue")
            # 图像增强处理
            enhanced = self._process_image(original_img)
            # 将OpenCV格式的图片转换为PIL格式以便添加水印
            enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            # 调用_add_watermark方法添加水印
            self._add_watermark(enhanced_pil)
            # 生成带时间戳的保存文件名
            filename = f"enhanced_{time.strftime('%Y%m%d%H%M%S')}.jpg"
            # 拼接完整保存路径
            self.sharpened_path = os.path.join(self.base_dir, filename)
            # 保存处理后的图片
            enhanced_pil.save(self.sharpened_path)
            # 显示增强后的图片
            self.show_image(self.sharpened_path)
            self.status_bar.config(text="图像增强完成", foreground="blue")
        except Exception as e:
            messagebox.showerror("处理错误", str(e))

    def _process_image(self, img):
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 计算图像的拉普拉斯方差，用于检测图像的清晰度
        sharpen = cv2.Laplacian(gray, cv2.CV_64F).var()
        # 根据清晰度选择不同的处理方法
        if sharpen < 100:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            enhanced = cv2.filter2D(img, -1, kernel)
        else:
            enhanced = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
        # 改变背景颜色
        # 将增强后的图像从BGR格式转换为HSV格式
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

        # H(色调):0-180(OpenCV中H范围是0-180而非0-360)
        # S(饱和度):0-255(0表示灰度,255表示完全饱和)
        # V(亮度):0-255(0表示黑色,255表示白色)
        # 设置H为0-180(任意色调),S为0-30(低饱和度),V为200-255(高亮度)为白色的下限值
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        # 设置H为0-180(任意色调),S为30(低饱和度),V为255(最高亮度)为白色的上限值
        upper_white = np.array([180, 30, 255], dtype=np.uint8)
        # 生成掩膜：在HSV颜色空间中检测白色区域
        mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # 创建浅蓝色背景图像
        # np.full_like创建与enhanced相同形状的数组
        background = np.full_like(enhanced, (255, 200, 200), dtype=np.uint8)
        
        # 背景替换：
        # 使用np.where将掩膜区域替换为背景色background
        # mask[...,None]增加一个维度以匹配enhanced的通道数
        enhanced = np.where(mask[..., None], background, enhanced)
        return enhanced

    def _add_watermark(self, img_pil):
        try:
            # 创建一个透明的水印层，大小与原图相同
            watermark_layer = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
            # 创建绘图对象用于在水印层上绘制
            draw = ImageDraw.Draw(watermark_layer)

            # 设置水印文本内容
            watermark_text = "水印"
            # 指定字体文件路径
            font_path = "SimHei.ttf"
            # 设置字体大小
            font_size = 20
            # 设置字体颜色(RGBA格式)，30表示30%透明度
            font_color = (0, 0, 0, 30)  # 半透明

            try:
                # 加载指定字体
                font = ImageFont.truetype(font_path, font_size)
                # 计算水印文本的边界框
                bbox = draw.textbbox((0, 0), watermark_text, font=font)
                # 计算水印文本的宽度和高度
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                # 增加更多水印位置
                positions = []
                # 四角位置
                positions.extend([
                    (20, 20),  # 左上
                    (img_pil.size[0] - text_width - 20, 20),  # 右上
                    (20, img_pil.size[1] - text_height - 20),  # 左下
                    (img_pil.size[0] - text_width - 20, img_pil.size[1] - text_height - 20),  # 右下
                ])

                # 中心区域
                positions.extend([
                    ((img_pil.size[0] - text_width) // 2, (img_pil.size[1] - text_height) // 2),  # 正中心
                    ((img_pil.size[0] - text_width) // 3, (img_pil.size[1] - text_height) // 3),  # 左上1/3处
                    ((img_pil.size[0] - text_width) * 2 // 3, (img_pil.size[1] - text_height) // 3),  # 右上1/3处
                    ((img_pil.size[0] - text_width) // 3, (img_pil.size[1] - text_height) * 2 // 3),  # 左下1/3处
                    ((img_pil.size[0] - text_width) * 2 // 3, (img_pil.size[1] - text_height) * 2 // 3),  # 右下1/3处
                ])
                # 边缘位置
                step = img_pil.size[0] // 8
                for i in range(1, 8):
                    positions.extend([
                        (i * step, 20),  # 上边缘
                        (i * step, img_pil.size[1] - text_height - 20),  # 下边缘
                        (20, i * img_pil.size[1] // 8),  # 左边缘
                        (img_pil.size[0] - text_width - 20, i * img_pil.size[1] // 8),  # 右边缘
                    ])
                # 遍历所有计算好的位置，添加水印文本
                for x, y in positions:
                    draw.text((x, y), watermark_text, font=font, fill=font_color)
                print("水印添加完成")

                # 添加斜向水印，从左上到右下和从右上到左下
                for i in range(0, img_pil.size[0] + img_pil.size[1], 100):  # 每100像素添加一个水印
                    draw.text((i, 0), watermark_text, font=font, fill=font_color)  # 从顶部开始斜向下
                    draw.text((0, i), watermark_text, font=font, fill=font_color)  # 从左侧开始斜向下
                # 对水印层应用高斯模糊，使水印更自然
                watermark_layer = watermark_layer.filter(ImageFilter.GaussianBlur(1.5))
                # 将水印层合并到原始图像上
                img_pil.paste(watermark_layer, (0, 0), watermark_layer)
            except Exception as e:
                print(f"添加水印失败: {str(e)}")
        except Exception as e:
            print(f"水印处理错误: {str(e)}")

    def run_ocr(self):
        if not self.sharpened_path:
            messagebox.showwarning("错误", "请先进行图像处理")
            return

        if not self.ocr_ready:
            messagebox.showinfo("初始化中",
                                f"当前加载进度：{self.models_loaded}/3 个模型\n请稍候...")
            return

        def ocr_task():
            try:
                # 更新状态栏显示识别中状态
                self.status_bar.config(text="正在识别...", foreground="orange")
                
                # 第一次识别：使用OCR引擎处理增强后的图像
                # result结构: [[[文本框坐标], (识别文本, 置信度)], ...]
                result = self.ocr_engine.ocr(self.sharpened_path)
                # 提取所有识别文本
                texts = [line[1][0] for line in result[0]]
                # 提取所有置信度分数
                confidences = [line[1][1] for line in result[0]]
                # 将识别结果合并为多行文本
                self.ocr_result = "\n".join(texts)
                
                # 生成热力图可视化识别结果
                # 读取增强后的图像
                img = cv2.imread(self.sharpened_path)
                # 创建图像副本用于叠加效果
                overlay = img.copy()
                # 创建与图像相同大小的空白掩膜
                mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)
                # 遍历每个识别结果，在掩膜上标记高置信度区域
                for line in result[0]:
                    # 获取文本框坐标并转换为整数类型
                    box = np.array(line[0]).astype(np.int32)
                    # 获取当前文本的置信度
                    confidence = line[1][1]
                    # 只处理置信度高于0.2的区域
                    if confidence > 0.2:
                        # 在掩膜上填充文本框区域为白色(255)
                        cv2.fillPoly(mask, [box], 255) 
                # 创建白色半透明颜色层
                color_layer = np.zeros_like(img, dtype=np.uint8)
                color_layer[:] = (255, 255, 255)
                # 设置叠加透明度
                alpha = 0.5
                # 将颜色层与原图按透明度混合
                temp = cv2.addWeighted(color_layer, alpha, overlay, 1 - alpha, 0)
                # 使用掩膜只保留高置信度区域的混合效果
                overlay = cv2.bitwise_and(temp, temp, mask=mask)
                # 合并处理后的图像和原始图像
                overlay = cv2.add(overlay, cv2.bitwise_and(overlay, overlay, mask=cv2.bitwise_not(mask)))
                # 保存热力图图像
                overlay_path = os.path.join(self.base_dir, f"overlay_{time.strftime('%Y%m%d%H%M%S')}.jpg")
                cv2.imwrite(overlay_path, overlay)
                # 对热力图进行图像增强处理
                enhanced_overlay = self._process_image(overlay)
                # 保存增强后的热力图
                enhanced_overlay_path = os.path.join(self.base_dir,f"enhanced_overlay_{time.strftime('%Y%m%d%H%M%S')}.jpg")
                cv2.imwrite(enhanced_overlay_path, enhanced_overlay)
                # 第二次识别
                second_result = self.ocr_engine.ocr(enhanced_overlay_path)
                # 提取第二次识别的文本结果
                second_texts = [line[1][0] for line in second_result[0]]
                # 以第二次识别结果为准
                self.ocr_result = "\n".join(second_texts)
                # 更新UI显示结果
                self.result_text.delete(1.0, tk.END)
                # 插入新的识别结果
                self.result_text.insert(tk.END, self.ocr_result)
                # 显示增强后的热力图
                self.show_image(enhanced_overlay_path)
                # 更新状态栏显示完成状态
                self.status_bar.config(text="二次识别完成", foreground="green")
                
                # 插入处理信息到imageprocess表
                with connection.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) AS count FROM imageprocess")
                    result = cursor.fetchone()
                    process_id = result['count'] + 1
                    
                    sql = """
                    INSERT INTO imageprocess (process_id, image_path, enhanced_path, process_time, result_summary)
                    VALUES (%s, %s, %s, %s, %s)
                    """
                    cursor.execute(sql, (
                        process_id,
                        self.sharpened_path,
                        enhanced_overlay_path,
                        time.strftime('%Y-%m-%d %H:%M:%S'),
                        self.ocr_result
                    ))
                    connection.commit()
            except Exception as e:
                messagebox.showerror("识别错误", str(e))
                self.status_bar.config(text="识别失败", foreground="red")
                
                # 记录错误日志到数据库
                with connection.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) AS count FROM systemlog")
                    result = cursor.fetchone()
                    log_id = result['count'] + 1
                    
                    sql = """
                    INSERT INTO systemlog (log_id, log_time, log_level, log_message)
                    VALUES (%s, %s, %s, %s)
                    """
                    cursor.execute(sql, (
                        log_id,
                        time.strftime('%Y-%m-%d %H:%M:%S'),
                        1,  # 固定日志级别为1
                        str(e)
                    ))
                    connection.commit()

        threading.Thread(target=ocr_task, daemon=True).start()

    def generate_report(self):
        if not self.ocr_result:
            messagebox.showwarning("错误", "没有可保存的识别结果")
            return

        try:
            pdf = FPDF()
            pdf.add_page()
            # 添加支持中文的 simhei.ttf 字体文件
            pdf.add_font('SimHei', '', 'SimHei.ttf', uni=True)
            pdf.set_font('SimHei', size=12)
            # 插入文本坐标
            pdf.set_xy(110, 20)
            # 输出多行文本
            # multi_cell(w, h, txt, align)参数说明:
            # 自动换行宽度(0表示使用页面右边界)w=0
            # 行高h=10
            # 要输出的文本内容txt
            # 左对齐align='L'
            pdf.multi_cell(0, 10, self.ocr_result, align='L')
            # 保存PDF
            # 生成时间戳作为文件名
            formatted_time = time.strftime("%Y_%m_%d %H_%M_%S")
            report_path = f'{formatted_time}.pdf'
            # 保存PDF文件
            pdf.output(report_path)
            messagebox.showinfo("成功", f"PDF报告已生成：\n{report_path}")
        except Exception as e:
            messagebox.showerror("PDF错误", str(e))

    def translate_text(self):
        if not self.ocr_result:
            messagebox.showwarning("错误", "没有可翻译的文本")
            return
        try:
            self.status_bar.config(text="正在翻译...", foreground="orange")
            # 使用大模型进行翻译
            chain = prompt | llm
            # 调用链并传入文本
            translated = chain.invoke({"text": self.ocr_result})
            # 显示结果
            # 清空文本框
            self.result_text.delete(1.0, tk.END)
            # 插入原始文本
            self.result_text.insert(tk.END, "=== 原始文本 ===\n")
            self.result_text.insert(tk.END, self.ocr_result + "\n\n")
            # 插入翻译结果
            self.result_text.insert(tk.END, "=== 翻译结果 ===\n")
            self.result_text.insert(tk.END, translated.content)
            self.status_bar.config(text="翻译完成", foreground="green")
        except Exception as e:
            messagebox.showerror("翻译错误", f"翻译失败: {str(e)}")
            self.status_bar.config(text="翻译失败", foreground="red")

    def _is_chinese(self, text):
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False

    def show_image(self, path):
        try:
            img = Image.open(path)
            img.thumbnail((800, 600))
            photo = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
        except Exception as e:
            messagebox.showerror("图像错误", str(e))

    def on_close(self):
        if messagebox.askokcancel("退出", "确定要退出吗？"):
            self._shutdown()

    def _shutdown(self):
        """安全关闭所有资源"""
        if self.progress:
            self.progress.stop()
        if self.ocr_engine:
            del self.ocr_engine
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()
