import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

imp_path = "Koalainputimage.jpg"
img = Image.open(imp_path)

img_np = np.array(img)
plt.imshow(img_np)
plt.show()

import torchvision
from torchvision import transforms
import torch
from torch.utils.data import DataLoader

# araç nesne olarak oluşturuldu
#transform = transforms.ToTensor() # transforms.ToTensor() bir class
# nesne tensor formunda değil
#img_tr = transform(img)
#mean, std = torch.mean(img_tr), torch.std(img_tr)

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# (0.5,), (0.5,) : elindeki görsel seti siyah beyaz ise bunu kullanırsın

# img_normalized = transforms(img)
# plt.imshow(img_normalized)
# img_np = np.array(img_normalized)

train_dataset = torchvision.datasets.FashionMNIST(root='FashionMNIST', train=True, transform=transforms)
test_datasets = torchvision.datasets.FashionMNIST(root='FashionMNIST', train=False, transform=transforms)

#img_normalized_transpose = img_normalized.transpose(1, 2, 0)
#plt.imshow(img_normalized_transpose)

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_datasets, batch_size = 64, shuffle = True)

iter(train_loader) # veriyi batch'Ler halinde çekmeye uygun hale getirir. her batch te bir çift veri döner görsel ve etiketi
for img_batch, label_batch in iter(train_loader):
    img = img_batch[0].squeeze() # img = [1,28,28], plt.imshow([28,28])
    label = label_batch[0].item() #tensörün içinde tek bir sayı varsa item ile int yada float yapılabilir
    print(label)
    if label == 9:
        break
    plt.imshow(img)
    plt.show()

from torch import nn

class FashionClassifier(nn.Module):
    def __init__(self):
        super(FashionClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self,x):
        return self.layers(x)

model = FashionClassifier()
print(model)
img_r, label_r = next(iter(train_loader))
result = model(img_r)
print(result)
print(result.shape)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_model(epochs):
    for epoch in range(epochs):
        for img,lbl in train_loader:
            optimizer.zero_grad() # default: pytorch eski adımlardan kalan türevleri üst üste toplar
            # hafızayı temizlemek için sıfırlama emri vermelisin
            yhat = model(img)
            loss = criterion(yhat, lbl)
            # modelin tahminlerini ile gerçek etiketleri kayıp fonka vererek ceaza puanını açıklar
            loss.backward()
            # geriye doğru tüm katmanların türevlerini otomatik olarak hesapla
            optimizer.step()
            # türevleri ve öğrenme hızını kullanarak ağırlık matrisilerini günceller
            # türev fonkun değişim oranıdır ve grafiğin üzerindeki teğetin eğimidir.
            running_loss =+ loss.item()
            print(f"Epoch {epoch+1} - Loss: {running_loss/ len(train_loader)}")


train_model(epochs = 100)
model.eval() # sen artık eğitmeyeceğim sadece test ediyorum ağırlıkları dondur mesajı

correct = 0
total = 0

with torch.no_grad():  # ??????????
    for i_batch, l_batch in test_loader:
        outputs = model(i_batch)
        value, index = torch.max(outputs, 1)  # ???????????????

        # 3. Toplam resim sayısını biriktir
        total += label_batch.size(0) # ?????????????????

        # 4. Doğru tahminleri say ve biriktir
        correct += (index == label_batch).sum().item()

    print(f"Modelin Test Setindeki Başarı Oranı (Accuracy): %{100 * correct / total:.2f}")



