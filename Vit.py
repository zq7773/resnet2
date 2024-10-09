import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from transformers import DeiTForImageClassification, DeiTFeatureExtractor
# from transformers import CaiTForImageClassification, CaiTFeatureExtractor
# from transformers import ConvNextForImageClassification, ConvNextFeatureExtractor
from transformers import ConvNextForImageClassification, ConvNextImageProcessor
# from transformers import T2TViTForImageClassification, T2TViTFeatureExtractor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# # 加载ViT模型和特征提取器
# feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=1, ignore_mismatched_sizes=True) # 忽略大小不匹配 # 二分类任务
# model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# # 使用DeiT模型和特征提取器
# feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224')
# model = DeiTForImageClassification.from_pretrained('facebook/deit-base-distilled-patch16-224', num_labels=1)
# model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# # CaiT模型和特征提取器
# feature_extractor = CaiTFeatureExtractor.from_pretrained('facebook/cait-xxs24-224')
# model = CaiTForImageClassification.from_pretrained('facebook/cait-xxs24-224', num_labels=1)
# model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# # ConvNext模型和特征提取器
# processor = ConvNextImageProcessor.from_pretrained('facebook/convnext-tiny-224')
# model = ConvNextForImageClassification.from_pretrained('facebook/convnext-tiny-224', num_labels=1, ignore_mismatched_sizes=True)

# model.save_pretrained('./local_model')
# processor.save_pretrained('./local_processor')
# 从本地加载模型和处理器
model = ConvNextForImageClassification.from_pretrained('./local_model')
processor = ConvNextImageProcessor.from_pretrained('./local_processor')

model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# # T2T-ViT模型和特征提取器
# feature_extractor = T2TViTFeatureExtractor.from_pretrained('google/t2t-vit-14')
# model = T2TViTForImageClassification.from_pretrained('google/t2t-vit-14', num_labels=1)
# model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
# 自定义数据集类，使用feature_extractor进行图像预处理
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
            image = self.transform(images=image)['pixel_values'][0]

        return image, label

# 创建数据集和数据加载器，使用ViT特征提取器
transform = processor
train_dataset = ImageDataset('train', 'train.txt', transform)
val_dataset = ImageDataset('valid', 'val.txt', transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 二分类的损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练和验证
train_losses, train_accuracies = [], []
num_epochs = 150

for epoch in range(num_epochs):
    model.train()
    total_loss, correct_train, total_train = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to('cuda'), labels.to('cuda').view(-1, 1)

        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = torch.sigmoid(outputs) > 0.5
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(avg_loss)
    train_accuracies.append(train_accuracy)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

    # 验证
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to('cuda'), labels.to('cuda').view(-1, 1)
            outputs = model(images).logits
            predicted = torch.sigmoid(outputs) > 0.5
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f'Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

# 绘制训练曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
