from jinja2 import optimizer
from torch_geometric.datasets import Amazon
# birlikte sıkça satın alınma ilişkisi
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# kişilerin birbiriyle olan ilişkisini inceliyoruz o yüzden kenarları böleceğiz

from torch_geometric.transforms import RandomLinkSplit

dataset = Amazon(root = "/tmp/Amazon", name = "Computers")
data = dataset[0]

# RandomLinkSplit : edge-level random split into train, validation and test data

transform = RandomLinkSplit(num_val = 0.05, num_test = 0.15,is_undirected = True)
# edges are undirected
# transform f(x) gibi bir fonksiyon
train_data, val_data, test_data = transform(data)

print("num of edges :", data.edge_index.shape[1])
# print("num of edges in train data :", train_data.num_edges.shape[1])
print(dataset.num_classes)
print("dataset num of features: ", dataset.num_features)

# 16 boyutlu vektör yayıncının karakteri 16 sayıya indirgenir
# 2 boyuta indirgemeye çalışsaydık o kadar özellik iki boyuta sığmazdı
# 500 boyut olsaydı algoritma ortak bağlantıları yakalamak yerine her şeyi ezberlerdi

class AmazonGCNEncoder(torch.nn.Module):
    def __init__(self):
        super(AmazonGCNEncoder, self).__init__()
        self.first_layer = GCNConv(dataset.num_features, 64)
        self.second_layer = GCNConv(64, 16)

    def forward(self,x, edge_index):
        x = self.first_layer(x,edge_index)
        # edge_index sparse komşuluk matrisi
        x = F.relu(x)
        x = self.second_layer(x,edge_index)
        return x


model = AmazonGCNEncoder()

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# grafı matematiksel olarak tanımlayan iki unsur var:
# düğümler matrisi: düğümlerin özelliklerini tutan matris
# komşuluk matrisi : hangi düğümün hangisine bağlı olduğunu gösteren yapı

# bunu kullanmazsak optimizasyon motoru ürünlerin özelliklerini öğrenmek yerine
# embedding collapse (vektör çöküşü) tüm ürünleri uzayda tek bir noktaya üst üste yığar
# bu hata yapıldığında ceza kesilmesi için

print(data.edge_index)
print(train_data.edge_label_index)

def decode(z, edge_label_index):
    neighbor_nodes1 = edge_label_index[0]
    neighbor_nodes2 = edge_label_index[1],

    v_a = z[neighbor_nodes1]
    v_b = z[neighbor_nodes2]

    return (v_a * v_b).sum(dim=-1) # skaler çarpım

from torch_geometric.utils import negative_sampling

for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1)
    )
    # bunu kullanmazsak optimizasyon motoru ürünlerin özelliklerini öğrenmek yerine
    # embedding collapse (vektör çöküşü) tüm ürünleri uzayda tek bir noktaya üst üste yığar
    # bu hata yapıldığında ceza kesilmesi için

    edge_label_index = torch.cat([train_data.edge_label_index, neg_edge_index], dim=-1)

    real_answers = torch.ones(train_data.edge_label_index.size(1)) # size(1) : num of columns
    # [1,1,1,...]
    fake_answers = torch.zeros(neg_edge_index.size(1))
    # [0,0,0,...]
    target_matrix = torch.cat([real_answers, fake_answers], dim=0).to(train_data.x.device)
    # ikisini uç uca ekler [1,1,...,0,0,...]
    # train_data.x düğümlerin özellik matrisi

    z = model(train_data.x, train_data.edge_index)
    predictions = decode(z, edge_label_index)

    loss = criterion(predictions, target_matrix)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} -> Güncel Hata (Loss): {loss.item()}")







