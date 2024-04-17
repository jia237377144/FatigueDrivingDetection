import streamlit as st
import random  # 导入random模块，用于生成随机数
import os
import cv2
from PIL import Image
import pandas as pd
import time  # 导入time模块，用于处理时间
from label_name import Label_list
import torch
from ultralytics import YOLO
from torchvision import transforms
import matplotlib.pyplot as plt

cls_name = Label_list  # 定义类名列表
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(cls_name))]  # 为每个目标类别生成一个随机颜色

# 选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载模型
model = YOLO('./weights/best.pt')
model.to(device)
# model.eval()

# 定义预处理步骤
# preprocess = transforms.Compose([
#     transforms.Resize((640, 640)),  # YOLOv8通常使用640x640作为输入大小
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet均值和标准差
# ])

st.title("基于YOLOv8的疲劳驾驶检测")

# 创建一个文件夹用于保存上传的文件
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# 上传文件
def uploads_file(type):
    if type == "img":
        uploaded_img = st.file_uploader("请上传一张图片", type=["png", "jpg", "jpeg"])
        if uploaded_img is not None:
            # 使用 st.image 显示上传的图片
            image = Image.open(uploaded_img)
            st.image(image)
            # 加载并预处理图像
            # image = Image.open(uploaded_img)
            img_path = f"./test_media/{uploaded_img.name}"
            image.save(img_path)
            results = model.predict(source=img_path,
            save=True, 
            verbose=True, 
            save_txt=True,
            save_crop=True, 
            # visualize=True
            )
            image.close()
            predict_image=Image.open(f"{results[0].save_dir}/{uploaded_img.name}")
            st.image(predict_image)

            # with torch.no_grad():  # 不需要计算梯度，节省计算资源
            #     results = model(image_tensor)  # 运行预测

            print(f"results: {results}")
            # 解析结果并应用NMS
            # outputs = results.xyxy[0]  # 获取预测结果的边界框、类别和置信度
            # outputs = non_max_suppression(outputs, conf_thres=0.4, iou_thres=0.5)  # 应用NMS
            #
            # # 可视化边界框和类别
            # plt.imshow(image)
            # for *xyxy, conf, cls in outputs:
            #     label = f'{classes[int(cls)]} {conf:.2f}'  # 获取类别标签和置信度
            #     plot_one_box(xyxy, image, label=label, color=colors(int(cls), True))  # 绘制边界框和标签
            # plt.show()

            # print("推理时间: %.2f" % use_time)  # 打印预测所用的时间
    elif type == "video":
        uploaded_video = st.file_uploader("请上传一个视频", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            # 使用 st.video 显示上传的视频
            st.video(uploaded_video)
        # else:
            # 如果没有文件被上传，显示一个消息
            # st.write("请上传一个视频")
    else:
        st.error("请上传正确的文件类型")


# 摄像头
def camera():
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
            st.write("Video capture has ended.")
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame,channels="RGB")
        # 关闭摄像头
        if stop:
            break
    cap.release()

# 创建一个侧边栏菜单
st.sidebar.title('疲劳驾驶检测')
# st.sidebar.success("请选择检测方式。")


# 左侧栏目
option1 = st.sidebar.radio('', ('选择图片', '选择视频', '摄像头'))
# 根据用户的选择执行不同的操作
if option1 == '选择图片':
    uploads_file(type="img")
elif option1 == '选择视频':
    uploads_file(type="video")
elif option1 == '摄像头':
    # st.title("摄像头检测疲劳驾驶")
    # 使用列进行布局
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start = st.button("打开摄像头")
    with col2:
        stop = st.button("关闭摄像头")
    with col3:
        pass
    with col4:
        pass
    # 初始化显示图片的变量
    show_image = True

    # 打开摄像头
    if start:
        camera()
        show_image = False  # 点击按钮后隐藏图片
    # 关闭摄像头
    if stop:
        show_image = True  # 点击按钮后显示图片

    # 加载图片
    # image = Image.open('./image/camera.jpg')
    image = Image.open('./image/camera1.png')
    # 根据show_image的值来决定是否显示图片
    if show_image:
        st.image(image, use_column_width=True)
else:
    # 其他操作
    pass


# 显示结果数据，创建一个简单的Pandas DataFrame
data = {'画面标识': ['张三', '李四', '王五'],
        '结果': [25, 30, 35],
        '位置': ['北京', '上海', '深圳'],
        '置信度': ['北京', '上海', '深圳'],
        }
df = pd.DataFrame(data)

# 使用st.table显示DataFrame
# st.table(df)

# 将 DataFrame 转换为 HTML 表格，并添加居中样式
html_table = df.to_html(
    index=False,  # 不显示索引
    classes='streamlit-table-center',  # 自定义 CSS 类
    escape=False  # 允许 HTML 标签
)

# 自定义 CSS 样式，设置表格文字居中并自适应宽度
centered_css = """  
<style>  
.streamlit-table-center {  
    margin-left: auto;  
    margin-right: auto;  
    border-collapse: collapse;  
    width: 100%; /* 表格宽度自适应 */  
}  
.streamlit-table-center th, .streamlit-table-center td {  
    text-align: center; /* 文字居中 */  
    # border: 1px solid black; /* 可选：添加边框以更好地看到表格结构 */  
}  
</style>  
"""

# 在 Streamlit 中显示带有自定义样式的 HTML 表格
st.markdown(centered_css, unsafe_allow_html=True)
st.markdown(html_table, unsafe_allow_html=True)