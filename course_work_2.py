# подключаем библиотеки
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt   


# Реализация активации HeLU.
class HELU(torch.autograd.Function):
    @staticmethod
    # Прямой проход 
    def forward(ctx, z, alpha):
        relu = torch.where(z > 0, z, torch.zeros_like(z))
        ctx.save_for_backward(z)
        ctx.alpha = alpha
        return relu

    @staticmethod
    # Обратный проход
    def backward(ctx, grad_output):
        z = ctx.saved_tensors[0]
        alpha = ctx.alpha
        grad_positive = torch.ones_like(z)
        grad_helu = torch.where(z > -alpha, grad_positive, torch.zeros_like(z))
        return grad_helu * grad_output, None


class HELUActivation(nn.Module):
    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return HELU.apply(x, self.alpha)


# простая свёрточная сеть.
# 2 свёрточных слоя + 1 полносвязный слой.
class CNN(nn.Module):
    def __init__(self, activation="relu"):
        super().__init__()
        def make_act():
            if activation == "relu":
                return nn.ReLU()
            else:
                return HELUActivation(alpha=0.05)

        # свёрточная часть
        # 3 -> 32 канала, ядро 3x3, padding=1 
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.act1 = make_act()
        # 32 -> 64 канала, ядро 3x3, padding=1 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.act2 = make_act()
        # общий слой подвыборки 2x2: 32 -> 16 -> 8 
        self.pool = nn.MaxPool2d(2, 2)

    
        self.dropout = nn.Dropout(0.25)

        # полносвязная часть - 1 слой
        self.fc = nn.Linear(64 * 8 * 8, 10)
        self._init_weights()


    def _init_weights(self):
        # Kaiming uniform
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv2.bias)
        # Xavier uniform
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.pool(self.act1(self.conv1(x)))  
        x = self.pool(self.act2(self.conv2(x)))  
        x = x.view(x.size(0), -1)                 # разворачиваем в вектор
        x = self.dropout(x)
        x = self.fc(x)                            # классификатор на 10 классов
        return x


# обучаем модель 
def train(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()    # обнуляем градиенты
        pred = model(x)          # расчитываем предсказания модели
        loss = loss_fn(pred, y)  # вычисляем ошибку
        loss.backward()     # обратное распространение ошибки
        optimizer.step()    # шаг оптимизации

        pred_labels = pred.argmax(dim=1)
        correct += (pred_labels == y).sum().item()
        total += y.size(0)

    return correct / total * 100  # точность на обучении


# проход модели на тестовых данных 
def test(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred_labels = pred.argmax(dim=1)
            correct += (pred_labels == y).sum().item()
            total += y.size(0)

    return correct / total * 100


# нормализация 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

# скачиваем набор данных и делим на обучающую и тестовую выборки
train_data = datasets.CIFAR10(root=".", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root=".", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

# используем GPU, если он доступен 
device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.CrossEntropyLoss()   # выбираем тип ошибки

EPOCHS = 30

results = {}   # сохраняем результаты
history = {}   # сохраняем кривые обучения по эпохам

for activation in ["relu", "helu"]:
    print(f"\n Activation function: {activation} ")

    model = CNN(activation=activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_accuraces = []   #  точность на обучении по эпохам
    test_accuraces = []    #  точность на тесте по эпохам

    # запускаем обучение
    for epoch in range(EPOCHS):
        train_accuracy = train(model, train_loader, optimizer, loss_fn, device)
        test_accuracy = test(model, test_loader, device)
        train_accuraces.append(train_accuracy)   # сохранение предсказаний на обучающей выборке
        test_accuraces.append(test_accuracy)     # сохранение предсказаний на тестовой выборке
        print(f"Epoch {epoch+1}: Train Accuracy = {train_accuracy:.2f}%, Test Accuracy = {test_accuracy:.2f}%")

    results[activation] = test_accuracy
    history[activation] = {"train": train_accuraces, "test": test_accuraces}   

# выводим итоговую точность для каждой модели по отдельности 
print("\nFinal Accuracy:", results)

# строим кривые обучения 
# Слева — точность на обучении, справа — на тесте

epochs_axis = range(1, EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# левый график — точность на обучающей выборке
ax1.plot(epochs_axis, history["relu"]["train"], label="ReLU", marker="o")
ax1.plot(epochs_axis, history["helu"]["train"], label="HeLU", marker="o")
ax1.set_title("Train Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Percent %")
ax1.legend()
ax1.grid(True)

# правый график — точность на тестовой выборке
ax2.plot(epochs_axis, history["relu"]["test"], label="ReLU", marker="o")
ax2.plot(epochs_axis, history["helu"]["test"], label="HeLU", marker="o")
ax2.set_title("Test Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Percent %")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150)   # сохраняем график в файл
plt.show()
