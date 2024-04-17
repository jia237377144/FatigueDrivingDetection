import torch
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载模型权重
model = YOLO('./weights/best-yolov8n.pt')
model.to(device)
model.eval()

# 定义预处理步骤
preprocess = transforms.Compose([
    transforms.Resize((640, 640)),  # YOLOv8通常使用640x640作为输入大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet均值和标准差
])

# 加载并预处理图像
image_path = './test_media/Drowsiness_SIXU_A00936.jpg'  # 替换为你的图像路径
# image = Image.open(image_path)
# image_tensor = preprocess(image).unsqueeze(0).to(device)  # 添加批次维度并移动到设备

with torch.no_grad():  # 不需要计算梯度，节省计算资源
    results = model(image_path)  # 运行预测

print(results)
# 解析结果并应用NMS
# outputs = results.xyxy[0]  # 获取预测结果的边界框、类别和置信度
# outputs = non_max_suppression(outputs, conf_thres=0.4, iou_thres=0.5)  # 应用NMS

# import matplotlib.pyplot as plt
#
# # 可视化边界框和类别
# plt.imshow(image)
# for *xyxy, conf, cls in outputs:
#     label = f'{classes[int(cls)]} {conf:.2f}'  # 获取类别标签和置信度
#     plot_one_box(xyxy, image, label=label, color=colors(int(cls), True))  # 绘制边界框和标签
# plt.show()