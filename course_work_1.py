import numpy as np
import torch
import random
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt 

# добавляем seed для исклбчения случайной инициализации
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

class ECGDataset(Dataset):
    def __init__(self, path):
        data = np.loadtxt(path, delimiter="\t")

        labels = data[:, 0].astype(int)
        unique = np.unique(labels)
        mapping = {c: i for i, c in enumerate(unique)}
        self.y = np.vectorize(mapping.get)(labels)

        self.x = data[:, 1:].astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])

train_dataset = ECGDataset("ECG5000_TRAIN.tsv")
test_dataset  = ECGDataset("ECG5000_TEST.tsv")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)


class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3):
        super().__init__()
        # Классическая рекуррентная сеть 
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh' # функция активации в рекуррентном слое

        )
        self.fc = nn.Linear(hidden_size, 5) 

    def forward(self, x):
        if x.dim() == 2:  # если признака нет — добавляем его
            x = x.unsqueeze(-1)       
        out, _ = self.rnn(x)
        last = out[:, -1, :]  # выход последнего временного шага       
        return self.fc(last)


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3):
        super().__init__()
        # Рекуррентный слой LSTM: хранит долговременную память в ячейке состояния
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,  # размер скрытого состояния
            num_layers=num_layers,    # число слоёв LSTM
            batch_first=True          # формат тензора
        )
        self.fc = nn.Linear(hidden_size, 5) # классификатор на 5 классов

    def forward(self, x):
        x = x.unsqueeze(-1)  # добавляем ось признака
        out, _ = self.lstm(x)  # выходы по всем временным шагам
        last = out[:, -1, :]   # берём выход последнего шага
        return self.fc(last)   # предсказание класса
    
    
def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()

        correct += (pred.argmax(1) == y).sum().item()
        total += y.size(0)

    return 100 * correct / total

def test_epoch(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)

    return 100 * correct / total


device = "cuda" if torch.cuda.is_available() else "cpu"

models = {
    "RNN":  RNN().to(device),
    "LSTM": LSTM().to(device)
}
EPOCHS = 30
results = {}   # сохраняем результаты
history = {}   # сохраняем кривые обучения по эпохам

for name, model in models.items():
    print("\n", name)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train_accuraces = []   #  точность на обучении по эпохам
    test_accuraces = []    #  точность на тесте по эпохам

    for epoch in range(EPOCHS):
        train_accuracy = train_epoch(model, train_loader, opt, loss_fn, device)
        test_accuracy  = test_epoch(model, test_loader, device)
        train_accuraces.append(train_accuracy)   # сохранение предсказаний на обучающей выборке
        test_accuraces.append(test_accuracy)     # сохранение предсказаний на тестовой выборке
        print(f"Epoch {epoch+1}: train={train_accuracy:.2f}% | test={test_accuracy:.2f}%")

    results[name] = test_accuraces[-1]
    history[name] = {"train": train_accuraces, "test": test_accuraces}

print("\nRESULTS:", results)

# строим кривые обучения 
# Слева — точность на обучении, справа — на тесте

epochs_axis = range(1, EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# левый график — точность на обучающей выборке
ax1.plot(epochs_axis, history["RNN"]["train"], label="RNN", marker="o")
ax1.plot(epochs_axis, history["LSTM"]["train"], label="LSTM", marker="o")
ax1.set_title("Train Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Percent %")
ax1.legend()
ax1.grid(True)

# правый график — точность на тестовой выборке
ax2.plot(epochs_axis, history["RNN"]["test"], label="RNN", marker="o")
ax2.plot(epochs_axis, history["LSTM"]["test"], label="LSTM", marker="o")
ax2.set_title("Test Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Percent %")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150)   # сохраняем график в файл
plt.show()