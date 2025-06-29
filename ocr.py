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

# 调用大模型
llm = ChatZhipuAI(
    zhipuai_api_key='c2307906dbf8486291c9b9ed6c46bc98.xcbSzlxvkmKMu7Gk',
    model="glm-4-flash",
    temperature=0.8,
    top_p=0.9,
)

# 构建模板
template = """
你是一个专业的翻译助手，你的任务是将输入的文本翻译成中文。
请将以下文本翻译成中文：
{text},并返回文本的重点"""
prompt = ChatPromptTemplate.from_template(template)

# 搭建链
chain = prompt | llm


class OCRApp:
    def __init__(self, master):
        self.master = master
        master.title("OCR专业版 v2.2")
        master.geometry("1000x800")
        master.protocol("WM_DELETE_WINDOW", self.on_close)

        # 初始化变量
        self.photo_path = ""
        self.sharpened_path = ""
        self.ocr_result = ""
        self.base_dir = 'wait_use'
        self.create_storage_dir()

        # OCR引擎状态
        self.ocr_engine = None
        self.ocr_ready = False
        self.models_loaded = 0
        self.init_success = False

        # 界面初始化（必须先创建控件）
        self.progress = None  # 显式声明属性
        self.create_widgets()  # 确保进度条先创建

        # 网络检测
        if self.check_network():
            self.init_ocr_engine()
        else:
            master.after(100, self._shutdown)

        # 摄像头资源
        self.cap = None
        self.is_capturing = False
        self.current_frame = None

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
            os.makedirs(self.base_dir, exist_ok=True)
            if not os.access(self.base_dir, os.W_OK):
                raise PermissionError("目录写入权限被拒绝")
        except Exception as e:
            self.base_dir = os.path.join(os.path.expanduser("~"), "Desktop")
            os.makedirs(self.base_dir, exist_ok=True)
            messagebox.showinfo("路径变更", f"存储路径已切换至桌面：{self.base_dir}")

    def init_ocr_engine(self):
        """改进的OCR引擎初始化"""

        def update_gui_status(msg, color="black"):
            self.master.after(0, lambda: self.status_bar.config(
                text=f"[{time.strftime('%H:%M')}] {msg}",
                foreground=color
            ))

        def load_models():
            try:
                # 组件有效性验证
                if self.progress is None or not self.progress.winfo_exists():
                    raise AttributeError("进度条组件异常")

                update_gui_status("正在初始化引擎...")
                self.progress.start(10)  # 启动进度条动画

                # Auto download models
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang='ch',
                    use_gpu=False,
                    enable_mkldnn=True,
                    show_log=True,
                    rec_model_dir=r'C:\Users\.paddleocr\whl\rec\ch\ch_PP-OCRv4_rec_infer',
                    cls_model_dir=r'C:\Users\.paddleocr\whl\cls\ch_ppocr_mobile_v2.0_cls_infer',
                    det_model_dir=r'C:\Users\.paddleocr\whl\det\ch\ch_PP-OCRv4_det_infer'

                )

                # 模拟加载过程反馈
                for stage in ["检测模型", "识别模型", "分类模型"]:
                    time.sleep(1)
                    self.models_loaded += 1
                    update_gui_status(f"正在加载{stage} ({self.models_loaded}/3)")

                # 预加载测试
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                self.ocr_engine.ocr(test_img)

                self.ocr_ready = True
                update_gui_status("OCR引擎就绪", "green")
                self.progress.stop()

            except Exception as e:
                self.progress.stop() if self.progress else None
                error_msg = f"""初始化失败！
                可能原因：
                1. 缺少VC++运行库（需安装2015+版本）
                2. 磁盘空间不足（需300MB以上）
                3. 防火墙阻止下载

                错误详情：{str(e)}"""
                self.master.after(0, lambda: messagebox.showerror(
                    "初始化失败", error_msg))
                update_gui_status("初始化失败", "red")
                self._shutdown()

        if self.progress:  # 确认组件已存在
            threading.Thread(target=load_models, daemon=True).start()
        else:
            self._shutdown()

    def create_widgets(self):
        """优化后的用户界面布局"""
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="操作面板", width=240)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # 显示区域
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 图像预览
        self.preview_label = ttk.Label(display_frame, background='#333')
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # 识别结果
        result_frame = ttk.LabelFrame(display_frame, text="识别结果")
        result_frame.pack(fill=tk.BOTH, expand=True)
        self.result_text = tk.Text(result_frame, wrap=tk.WORD, font=('微软雅黑', 10))
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 控制按钮
        btn_style = {'fill': tk.X, 'padx': 5, 'pady': 3}
        ttk.Button(control_frame, text="启动摄像头", command=self.toggle_camera).pack(**btn_style)
        ttk.Button(control_frame, text="拍摄照片", command=self.capture_image).pack(**btn_style)
        ttk.Button(control_frame, text="选择图片", command=self.select_image).pack(**btn_style)
        ttk.Button(control_frame, text="图像增强", command=self.enhance_image).pack(**btn_style)
        ttk.Button(control_frame, text="文字识别", command=self.run_ocr).pack(**btn_style)
        ttk.Button(control_frame, text="生成报告", command=self.generate_report).pack(**btn_style)
        ttk.Button(control_frame, text="翻译文本", command=self.translate_text).pack(**btn_style)

        # 进度条（关键修复点）
        self.progress = ttk.Progressbar(
            control_frame,
            mode='indeterminate',
            orient=tk.HORIZONTAL,
            length=200
        )
        self.progress.pack(fill=tk.X, pady=5)

        # 状态栏
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
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise IOError("无法访问摄像头")
            self.is_capturing = True
            self.capture_thread = threading.Thread(target=self.update_preview)
            self.capture_thread.start()
            self.status_bar.config(text="摄像头已开启", foreground="blue")
        except Exception as e:
            messagebox.showerror("摄像头错误", str(e))

    def stop_camera(self):
        self.is_capturing = False
        if self.cap:
            self.cap.release()
        self.status_bar.config(text="摄像头已关闭", foreground="gray")

    def update_preview(self):
        while self.is_capturing:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img.thumbnail((800, 600))
                photo = ImageTk.PhotoImage(img)
                self.preview_label.configure(image=photo)
                self.preview_label.image = photo

    def capture_image(self):
        if self.current_frame is None:
            messagebox.showwarning("警告", "请先开启摄像头")
            return

        try:
            filename = f"capture_{time.strftime('%Y%m%d%H%M%S')}.jpg"
            self.photo_path = os.path.join(self.base_dir, filename)
            cv2.imwrite(self.photo_path, self.current_frame)
            self.status_bar.config(text=f"已保存: {filename}", foreground="green")
            self.stop_camera()
            self.show_image(self.photo_path)
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def select_image(self):
        """选择本地图片"""
        file_path = filedialog.askopenfilename(
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.photo_path = file_path
            self.show_image(file_path)
            self.status_bar.config(text="已选择图片", foreground="blue")

    def enhance_image(self):
        if not self.photo_path:
            messagebox.showwarning("错误", "请先选择图片")
            return

        try:
            # 加载并显示原始图片
            original_img = cv2.imread(self.photo_path)
            if original_img is None:
                raise ValueError("无法读取图片文件")

            self.show_image(self.photo_path)
            self.status_bar.config(text="原始图片已加载", foreground="blue")

            # 图像增强处理
            enhanced = self._process_image(original_img)

            # 添加水印并保存
            enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            self._add_watermark(enhanced_pil)

            filename = f"enhanced_{time.strftime('%Y%m%d%H%M%S')}.jpg"
            self.sharpened_path = os.path.join(self.base_dir, filename)
            enhanced_pil.save(self.sharpened_path)

            # 显示增强后的图片
            self.show_image(self.sharpened_path)
            self.status_bar.config(text="图像增强完成", foreground="blue")

        except Exception as e:
            messagebox.showerror("处理错误", str(e))

    def _process_image(self, img):
        """内部使用的图像处理方法"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpen = cv2.Laplacian(gray, cv2.CV_64F).var()

        if sharpen < 100:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            enhanced = cv2.filter2D(img, -1, kernel)
        else:
            enhanced = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

        # 改变背景颜色
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([180, 30, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # 创建浅蓝色背景
        background = np.full_like(enhanced, (255, 200, 200), dtype=np.uint8)
        enhanced = np.where(mask[..., None], background, enhanced)

        return enhanced

    def _add_watermark(self, img_pil):
        try:
            watermark_layer = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(watermark_layer)

            watermark_text = "水印"
            font_path = "SimHei.ttf"
            font_size = 20
            font_color = (0, 0, 0, 30)  # 半透明

            try:
                font = ImageFont.truetype(font_path, font_size)
                bbox = draw.textbbox((0, 0), watermark_text, font=font)
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
                    ((img_pil.size[0] - text_width) // 2, (img_pil.size[1] - text_height) // 2),
                    ((img_pil.size[0] - text_width) // 3, (img_pil.size[1] - text_height) // 3),
                    ((img_pil.size[0] - text_width) * 2 // 3, (img_pil.size[1] - text_height) // 3),
                    ((img_pil.size[0] - text_width) // 3, (img_pil.size[1] - text_height) * 2 // 3),
                    ((img_pil.size[0] - text_width) * 2 // 3, (img_pil.size[1] - text_height) * 2 // 3),
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

                # 在所有位置添加水印
                for x, y in positions:
                    draw.text((x, y), watermark_text, font=font, fill=font_color)
                print("水印添加完成")

                # 添加斜向水印
                for i in range(0, img_pil.size[0] + img_pil.size[1], 100):
                    draw.text((i, 0), watermark_text, font=font, fill=font_color)
                    draw.text((0, i), watermark_text, font=font, fill=font_color)

                watermark_layer = watermark_layer.filter(ImageFilter.GaussianBlur(1.5))
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
                self.status_bar.config(text="正在识别...", foreground="orange")
                # 第一次识别
                result = self.ocr_engine.ocr(self.sharpened_path)
                texts = [line[1][0] for line in result[0]]
                confidences = [line[1][1] for line in result[0]]
                self.ocr_result = "\n".join(texts)

                # 生成热力图
                img = cv2.imread(self.sharpened_path)
                overlay = img.copy()
                mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)
                for line in result[0]:
                    box = np.array(line[0]).astype(np.int32)
                    confidence = line[1][1]
                    if confidence > 0.2:
                        cv2.fillPoly(mask, [box], 255)

                color_layer = np.zeros_like(img, dtype=np.uint8)
                color_layer[:] = (255, 255, 255)
                alpha = 0.5
                temp = cv2.addWeighted(color_layer, alpha, overlay, 1 - alpha, 0)
                overlay = cv2.bitwise_and(temp, temp, mask=mask)
                overlay = cv2.add(overlay, cv2.bitwise_and(overlay, overlay, mask=cv2.bitwise_not(mask)))

                overlay_path = os.path.join(self.base_dir, f"overlay_{time.strftime('%Y%m%d%H%M%S')}.jpg")
                cv2.imwrite(overlay_path, overlay)

                enhanced_overlay = self._process_image(overlay)
                enhanced_overlay_path = os.path.join(self.base_dir,
                                                     f"enhanced_overlay_{time.strftime('%Y%m%d%H%M%S')}.jpg")
                cv2.imwrite(enhanced_overlay_path, enhanced_overlay)

                second_result = self.ocr_engine.ocr(enhanced_overlay_path)
                second_texts = [line[1][0] for line in second_result[0]]
                self.ocr_result = "\n".join(second_texts)  # 以第二次识别结果为准

                # 显示最终结果
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, self.ocr_result)
                self.show_image(enhanced_overlay_path)
                self.status_bar.config(text="二次识别完成", foreground="green")

            except Exception as e:
                messagebox.showerror("识别错误", str(e))
                self.status_bar.config(text="识别失败", foreground="red")

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

            # 插入文本
            pdf.set_xy(110, 20)
            pdf.multi_cell(0, 10, self.ocr_result, align='L')

            # 保存 PDF
            formatted_time = time.strftime("%Y_%m_%d %H_%M_%S")
            report_path = f'{formatted_time}.pdf'
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

            # 使用已配置的GLM-4大模型进行翻译
            chain = prompt | llm
            translated = chain.invoke({"text": self.ocr_result})

            # 显示结果：上半部分原始文本，下半部分翻译结果
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "=== 原始文本 ===\n")
            self.result_text.insert(tk.END, self.ocr_result + "\n\n")
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
