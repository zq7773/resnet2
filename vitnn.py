import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import timm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'https://127.0.0.1:7890'
# 自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, folder_path, labels_file, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_labels = self.load_labels(labels_file)

    def load_labels(self, labels_file):
        labels = {}
        with open(labels_file, 'r') as f:
            for line in f:
                img_name, label = line.strip().split()
                labels[img_name] = float(label)
        return labels

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_name = list(self.image_labels.keys())[idx]
        img_path = os.path.join(self.folder_path, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.image_labels[img_name]

        if self.transform:
            image = self.transform(image)

        return image, label

# 数据增强和预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 创建数据集和数据加载器
train_dataset = ImageDataset('train', 'train.txt', transform)
val_dataset = ImageDataset('valid', 'val.txt', transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 构建ViT模型
model = timm.create_model('vit_base_patch16_224', pretrained=False)

# 修改分类头部，适应二分类任务
model.head = nn.Linear(model.head.in_features, 1)

# 将模型移动到GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 二分类的损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 记录训练损失和准确率
train_losses = []
train_accuracies = []
num_epochs = 150  # 训练轮数

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
        images, labels = images.to(device), labels.to(device).view(-1, 1)

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

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

    # 验证模型
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).view(-1, 1)
            outputs = model(images)
            predicted = torch.sigmoid(outputs) > 0.5
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # 计算评价指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    writer.add_scalar('val_accuracy', accuracy, epoch)
    writer.add_scalar('val_precision', precision, epoch)
    writer.add_scalar('val_recall', recall, epoch)
    writer.add_scalar('val_f1', f1, epoch)

    print(f'Validation Accuracy: {accuracy:.4f}')
    print(f'Validation Precision: {precision:.4f}')
    print(f'Validation Recall: {recall:.4f}')
    print(f'Validation F1 Score: {f1:.4f}\n')

    # 保存最佳模型
    if best_accuracy == -1 or accuracy > best_accuracy:
        best_accuracy = accuracy
        best_epoch = epoch
        torch.save(model.state_dict(), model_save_path + '/best_model.pth')

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
