import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ======================
# 1. 加载你训练好的最佳模型
# ======================
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 2)

model.load_state_dict(torch.load("best.pth", map_location="cpu"))
model.eval()

# ======================
# 2. 图像预处理
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

label_map = {0: "Cat", 1: "Dog"}

# ======================
# 3. 预测函数
# ======================
def predict(img_path):
    img = Image.open(img_path).convert("RGB")

    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()

    print("图片路径:", img_path)
    print("预测结果：", label_map[pred])
    return label_map[pred]


# ======================
# 4. 示例：在这里输入你的测试图片路径
# ======================
if __name__ == "__main__":
    test_img = "./test1.jpg"
    predict(test_img)
