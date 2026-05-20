import torch

neighbor = [[0,1,1],
            [1,0,1],
            [1,1,0]]

A = torch.tensor(neighbor)
I = torch.eye(3,3)
A_ = A + I  # matematiksel olarak simetrik üçgen graf
print(A_)

A_d1 = torch.sum(A_, dim = 1) # satır toplamları
print("A_d1" , A_d1)


D_1_2 = torch.diag(1.0 / torch.sqrt(A_d1))
print("1.0 / torch.sqrt(A_d1)" , 1.0 / torch.sqrt(A_d1))
print("D_1_2" , D_1_2)

# torch.diag(tensor, 0) - köşegenleri verir.

symmetric_normalization = D_1_2 @ A_ @ D_1_2
print("symmetric_normalization",symmetric_normalization)
# matristeki sayılar arasında çok fazla büyüklük farkı olabilir
# bilgi yayılırken popüler değerler devasa büyür(exploding gradients) modelin optimize olmasını engeller
# bu loss backward da çarpıldıkça büyüyen sayılar ( partial derivative de çarpma yapılıyor) learning rate
# ile çarpıldığında devasa adımlar atılmasına sebep olur

# direk D^-1 ile çarpılabilir A ama simetrikliği bozulur o yüzden D^-1/2 ile çarpılır

# veri sonsuz olursa bilgisayar işleyemez.

x = torch.randn(3,2) # 3 row 2 col
print(x)

W = torch.randn(2,4)

c = symmetric_normalization @ x @ W
result = torch.relu(c) # f(x) = max(0,x)
print(result)
# Graf sinir ağlarında katman sayısını çok artırdığımızda düğümlerin birbirine benzemesi problemine
# Over-smoothing (Aşırı Pürüzsüzleşme) denir ve bu,
# şu an canlı kanıtını gördüğün saf bir lineer cebir sonucudur.

from torch_geometric.datasets import Planetoid

#Bazı veri setlerinde (örneğin kimyasal molekül veri setlerinde) elinizde tek bir graf değil, binlerce farklı molekül grafı olur.
# Bu yüzden PyTorch Geometric kütüphanesi standart olarak tüm veri setlerini birer "Graf Listesi" hiyerarşisinde indirir.
# Cora veri seti ise tek bir devasa ağdan oluşur - dataset[0]

dataset = Planetoid(root = "/tmp/Cora", name = "Cora")
data = dataset[0]

print(f"Graf nesnesi: {data}")
print(f"Toplam Düğüm Sayısı: {data.num_nodes}")
print(f"Toplam Kenar Sayısı: {data.num_edges}")
print(f"Giriş Özellik Boyutu: {dataset.num_features}")
print(f"Hedef Sınıf Sayısı: {dataset.num_classes}")
# makale hangi branşa ait
print(data.x)
# 2708 tane 1433 boyutlu vektör
print(data.edge_index)
# bağlantı haritası
# edge_index = [
#     [  0,   0,   1,   3,  215],  # Kaynak düğüm (Nereden çıkıyor?) : atıf yapan makale
#     [ 21,  45,  21,   0,    2]   # Hedef düğüm  (Nereye gidiyor?)  : atıf yapılan
# ]
print("Matristeki Toplam Eleman Sayısı:", data.x.numel())
print("Sıfırdan Farklı (1 olan) Eleman Sayısı:", torch.sum(data.x == 1).item())

from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class Graf_artic(torch.nn.Module):
    def __init__(self):
        super(Graf_artic, self).__init__()
        self.first_layer = GCNConv(dataset.num_features,16)
        self.second_layer = GCNConv(16,dataset.num_classes)

    def forward(self, data):
        x = data.x
        edge = data.edge_index
        x = self.first_layer(x, edge)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # over fitting i engellemek için rastgele satırların yarısını düşürür
        # test sırasında self.training = false olur
        x = self.second_layer(x, edge)

        return x

model = Graf_artic()
print(model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# SGD klasik gradyan hesaplaması ile çalışır

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    if(epoch % 20 == 0):
        print(f"epoch : {epoch}, loss: {loss.item()}")

model.eval()
# artık nöronların %50 rastgele sıfırlanmaz

with torch.no_grad():# ağırlıklar güncellenmeyecek gradyan takibini bırak
    out = model(data)
    prediction = torch.argmax(out, dim=1)
    true_sum =(data.y[data.test_mask] == prediction[data.test_mask]).sum().item()
    true_sum2 = torch.sum(data.y[data.test_mask] == prediction[data.test_mask]).item()
    print(f"true_sum: {true_sum}")
    print(f"true_sum2: {true_sum2}")
    accuracy = true_sum / data.test_mask.sum().item()
    print(f"accuracy: {accuracy}")









