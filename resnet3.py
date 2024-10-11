import datetime
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# 加载标签文件并获取图像路径
def load_image_paths_and_labels(image_folder, labels_file):
    image_paths = []
    labels = []
    with open(labels_file, 'r') as f:
        for line in f:
            img_name, label = line.strip().split()
            image_paths.append(os.path.join(image_folder, img_name))
            labels.append(float(label))
    return image_paths, labels

# 加载图像路径和标签
image_paths, labels = load_image_paths_and_labels('CIR_images1', 'CIR_labels1.txt')

# 按照7:3比例划分数据集
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.3, random_state=42, stratify=labels
)

# 数据增强和预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 创建训练集和验证集数据加载器
train_dataset = ImageDataset(train_paths, train_labels, transform)
val_dataset = ImageDataset(val_paths, val_labels, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 构建ResNet-18模型
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 1)
)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 记录训练损失和准确率
train_losses = []
train_accuracies = []
num_epochs = 100

#tensorboard展示
best_accuracy = -1
train_start_time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
log_dir = './logs/' + train_start_time
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)
model_save_path = "./model/" + str(train_start_time)
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to('cuda'), labels.to('cuda').view(-1, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 计算训练准确率
        predicted = torch.sigmoid(outputs) > 0.5
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(avg_loss)
    train_accuracies.append(train_accuracy)

    #tensorboard
    writer.add_scalar('train_loss', avg_loss, epoch)
    writer.add_scalar('train_accuracy', train_accuracy, epoch)


    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

    # 验证模型
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to('cuda'), labels.to('cuda').view(-1, 1)
            outputs = model(images)
            predicted = torch.sigmoid(outputs) > 0.5
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # 计算评价指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    #tensorboard展示
    writer.add_scalar('val_accuracy', accuracy, epoch)
    writer.add_scalar('val_precision', precision, epoch)
    writer.add_scalar('val_recall', recall, epoch)
    writer.add_scalar('val_f1', f1, epoch)


    # 保存最佳模型
    if best_accuracy == -1 or accuracy > best_accuracy:
        best_accuracy = accuracy
        best_epoch = epoch
        torch.save(model.state_dict(), model_save_path + '/best_model.pth')

    print(f'Validation Accuracy: {accuracy:.4f}')
    print(f'Validation Precision: {precision:.4f}')
    print(f'Validation Recall: {recall:.4f}')
    print(f'Validation F1 Score: {f1:.4f}\n')

# 绘制训练损失和准确率
plt.figure(figsize=(12, 5))

# 绘制损失
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# 绘制准确率
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
