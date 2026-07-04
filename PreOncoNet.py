# ============================================================
# STEP 1 : Multi-Cancer Genomic Data Collection
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# Dataset Paths
# ------------------------------------------------------------

BREAST_DATASET = "/content/breast_dataset.csv"
LIVER_DATASET  = "/content/liver_dataset.csv"

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

def load_dataset(path):
    df = pd.read_csv(path)
    print(f"\nLoaded : {os.path.basename(path)}")
    print("Shape :", df.shape)
    return df

breast_df = load_dataset(BREAST_DATASET)
liver_df  = load_dataset(LIVER_DATASET)

# ------------------------------------------------------------
# Add Cancer Type
# ------------------------------------------------------------

breast_df["CancerType"] = "Breast"
liver_df["CancerType"]  = "Liver"

# ------------------------------------------------------------
# Merge Dataset
# ------------------------------------------------------------

data = pd.concat([breast_df, liver_df], ignore_index=True)

print("\nMerged Shape :", data.shape)

# ------------------------------------------------------------
# Find Target Column Automatically
# ------------------------------------------------------------

possible_targets = [
    "label",
    "Label",
    "class",
    "Class",
    "target",
    "Target",
    "Diagnosis",
    "diagnosis",
    "Cancer",
    "CancerType"
]

target_col = None

for col in possible_targets:
    if col in data.columns:
        target_col = col
        break

if target_col is None:
    target_col = data.columns[-1]

print("\nTarget Column :", target_col)

# ------------------------------------------------------------
# Missing Value Summary
# ------------------------------------------------------------

print("\nMissing Values")
print(data.isnull().sum())

# ------------------------------------------------------------
# Encode Target
# ------------------------------------------------------------

le = LabelEncoder()
data[target_col] = le.fit_transform(data[target_col])

# ------------------------------------------------------------
# Separate Features
# ------------------------------------------------------------

X = data.drop(columns=[target_col])
y = data[target_col]

# ------------------------------------------------------------
# Remove Non Numeric Columns
# ------------------------------------------------------------

for col in X.columns:
    if X[col].dtype == object:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# ------------------------------------------------------------
# Convert Numeric
# ------------------------------------------------------------

X = X.apply(pd.to_numeric, errors="coerce")

# ------------------------------------------------------------
# Train Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)

# ------------------------------------------------------------
# Save Intermediate Data
# ------------------------------------------------------------

X_train.to_csv("X_train_step1.csv", index=False)
X_test.to_csv("X_test_step1.csv", index=False)

pd.DataFrame(y_train).to_csv("y_train_step1.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test_step1.csv", index=False)

print("\nStep 1 Completed Successfully.")

# ============================================================
# STEP 2 : Genomic Data Preprocessing & Harmonization
# ============================================================

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# Load Step-1 Output
# ------------------------------------------------------------

X_train = pd.read_csv("X_train_step1.csv")
X_test  = pd.read_csv("X_test_step1.csv")

print("Train Shape :", X_train.shape)
print("Test Shape  :", X_test.shape)

# ============================================================
# 1. Missing Value Imputation
# (MissForest can be substituted by missingpy.MissForest if installed)
# ============================================================

print("\nImputing Missing Values...")

imputer = KNNImputer(n_neighbors=5)

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns
)

# ============================================================
# 2. Batch Effect Correction (Simple Centering Approximation)
# ============================================================

print("Removing Batch Effects...")

train_mean = X_train.mean()

X_train = X_train - train_mean
X_test = X_test - train_mean

# ============================================================
# 3. Variance Stabilizing Transformation
# ============================================================

print("Variance Stabilization...")

pt = PowerTransformer(method='yeo-johnson')

X_train = pd.DataFrame(
    pt.fit_transform(X_train),
    columns=X_train.columns
)

X_test = pd.DataFrame(
    pt.transform(X_test),
    columns=X_test.columns
)

# ============================================================
# 4. Robust Scaling
# ============================================================

scaler = StandardScaler()

X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)

X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)

# ============================================================
# 5. Robust PCA
# ============================================================

print("Running RPCA...")

cov = MinCovDet().fit(X_train)

mahal = cov.mahalanobis(X_train)

threshold = np.percentile(mahal,95)

mask = mahal < threshold

X_train = X_train.loc[mask].reset_index(drop=True)

print("Remaining Samples :", X_train.shape[0])

# ------------------------------------------------------------
# PCA Feature Compression
# ------------------------------------------------------------

n_comp = min(300, X_train.shape[1])

pca = PCA(n_components=n_comp, random_state=42)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("PCA Shape :", X_train_pca.shape)

# ============================================================
# Save Outputs
# ============================================================

pd.DataFrame(X_train_pca).to_csv(
    "X_train_step2.csv",
    index=False
)

pd.DataFrame(X_test_pca).to_csv(
    "X_test_step2.csv",
    index=False
)

print("\nStep-2 Completed Successfully")

# ============================================================
# STEP 3 : Pre-Mutational Genomic Fluctuation Learning (PGFL)
# ============================================================

import numpy as np
import pandas as pd
import pywt

from scipy.stats import ttest_ind
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------
# Load Step-2 Output
# ------------------------------------------------------------

X_train = pd.read_csv("X_train_step2.csv")
X_test = pd.read_csv("X_test_step2.csv")

y_train = pd.read_csv("y_train_step1.csv").values.ravel()

print("Train :", X_train.shape)

# ============================================================
# 1. Wavelet Features
# ============================================================

def wavelet_features(data):

    feat = []

    for row in data.values:

        coeff = pywt.wavedec(row, 'db4', level=3)

        vec = []

        for c in coeff:
            vec.extend([
                np.mean(c),
                np.std(c),
                np.max(c),
                np.min(c),
                np.median(c)
            ])

        feat.append(vec)

    return np.array(feat)

print("Extracting Wavelet Features...")

wave_train = wavelet_features(X_train)
wave_test = wavelet_features(X_test)

# ============================================================
# 2. Sample Entropy
# ============================================================

def sample_entropy(signal):

    signal = np.asarray(signal)

    std = np.std(signal)

    if std == 0:
        return 0

    return np.var(np.diff(signal)) / (std + 1e-8)

print("Computing Sample Entropy...")

entropy_train = np.array([
    sample_entropy(x)
    for x in X_train.values
]).reshape(-1,1)

entropy_test = np.array([
    sample_entropy(x)
    for x in X_test.values
]).reshape(-1,1)

# ============================================================
# 3. Differential Expression
# ============================================================

print("Differential Expression Analysis...")

classes = np.unique(y_train)

group1 = X_train[y_train==classes[0]]
group2 = X_train[y_train==classes[1]]

t_score,p_value = ttest_ind(
    group1,
    group2,
    axis=0,
    equal_var=False
)

diff_score = np.abs(t_score)

# repeat for every sample

diff_train = np.tile(diff_score,(len(X_train),1))
diff_test = np.tile(diff_score,(len(X_test),1))

# ============================================================
# 4. Adaptive Variance Score
# ============================================================

variance = np.var(X_train.values,axis=1).reshape(-1,1)

variance_test = np.var(X_test.values,axis=1).reshape(-1,1)

# ============================================================
# 5. Feature Fusion
# ============================================================

train_pgfl = np.concatenate([
    wave_train,
    entropy_train,
    variance,
    diff_train
],axis=1)

test_pgfl = np.concatenate([
    wave_test,
    entropy_test,
    variance_test,
    diff_test
],axis=1)

# ============================================================
# Normalize
# ============================================================

scaler = MinMaxScaler()

train_pgfl = scaler.fit_transform(train_pgfl)
test_pgfl = scaler.transform(test_pgfl)

print("PGFL Feature Shape :",train_pgfl.shape)

# ============================================================
# Save
# ============================================================

pd.DataFrame(train_pgfl).to_csv(
    "PGFL_train.csv",
    index=False
)

pd.DataFrame(test_pgfl).to_csv(
    "PGFL_test.csv",
    index=False
)

print("\nSTEP-3 COMPLETED SUCCESSFULLY")
# ============================================================
# STEP 4 : Multi-Omics Biological Knowledge Graph Construction
# ============================================================

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from node2vec import Node2Vec

# ------------------------------------------------------------
# Load PGFL Features
# ------------------------------------------------------------

X_train = pd.read_csv("PGFL_train.csv")
X_test  = pd.read_csv("PGFL_test.csv")

print("Input Shape :", X_train.shape)

# ============================================================
# Build Gene Correlation Graph
# ============================================================

print("\nBuilding Knowledge Graph...")

corr = np.corrcoef(X_train.T)

threshold = 0.70

G = nx.Graph()

num_nodes = corr.shape[0]

for i in range(num_nodes):
    G.add_node(i)

for i in range(num_nodes):
    for j in range(i+1,num_nodes):

        if abs(corr[i,j]) > threshold:
            G.add_edge(i,j,weight=float(corr[i,j]))

print("Nodes :",G.number_of_nodes())
print("Edges :",G.number_of_edges())

# ============================================================
# Node2Vec Embedding
# ============================================================

print("\nRunning Node2Vec...")

node2vec = Node2Vec(
    G,
    dimensions=128,
    walk_length=20,
    num_walks=100,
    workers=4
)

model = node2vec.fit(
    window=10,
    min_count=1,
    batch_words=64
)

node_embeddings = np.array([
    model.wv[str(i)]
    for i in range(num_nodes)
])

print("Node Embedding :",node_embeddings.shape)

# ============================================================
# Prepare Graph for GraphSAGE
# ============================================================

edge_index = np.array(list(G.edges())).T

edge_index = torch.tensor(edge_index,dtype=torch.long)

x = torch.tensor(node_embeddings,dtype=torch.float)

graph = Data(
    x=x,
    edge_index=edge_index
)

# ============================================================
# GraphSAGE
# ============================================================

class GraphSAGE(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = SAGEConv(128,256)
        self.conv2 = SAGEConv(256,128)

    def forward(self,data):

        x=data.x
        edge=data.edge_index

        x=self.conv1(x,edge)
        x=F.relu(x)

        x=self.conv2(x,edge)

        return x

device="cuda" if torch.cuda.is_available() else "cpu"

graph=graph.to(device)

net=GraphSAGE().to(device)

optimizer=torch.optim.Adam(
    net.parameters(),
    lr=0.001
)

# ============================================================
# GraphSAGE Training
# ============================================================

print("\nTraining GraphSAGE...")

for epoch in range(100):

    optimizer.zero_grad()

    out=net(graph)

    loss=((out-graph.x)**2).mean()

    loss.backward()

    optimizer.step()

    if epoch%10==0:
        print(epoch,loss.item())

graph_embedding=out.detach().cpu().numpy()

print("\nGraph Embedding Shape :",graph_embedding.shape)

# ============================================================
# Generate Sample Embedding
# ============================================================

train_embedding=np.dot(
    X_train.values,
    graph_embedding
)

test_embedding=np.dot(
    X_test.values,
    graph_embedding
)

print(train_embedding.shape)

# ============================================================
# Save Outputs
# ============================================================

pd.DataFrame(train_embedding).to_csv(
    "Graph_train.csv",
    index=False
)

pd.DataFrame(test_embedding).to_csv(
    "Graph_test.csv",
    index=False
)

print("\nSTEP 4 COMPLETED SUCCESSFULLY")

# ============================================================
# STEP 5 : Pathway-Guided Transformer Susceptibility Network
# ============================================================

import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset,DataLoader

# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------

pgfl_train=pd.read_csv("PGFL_train.csv").values
pgfl_test=pd.read_csv("PGFL_test.csv").values

graph_train=pd.read_csv("Graph_train.csv").values
graph_test=pd.read_csv("Graph_test.csv").values

y_train=pd.read_csv("y_train_step1.csv").values.ravel()
y_test=pd.read_csv("y_test_step1.csv").values.ravel()

# ------------------------------------------------------------
# Feature Fusion
# ------------------------------------------------------------

X_train=np.concatenate([pgfl_train,graph_train],axis=1)
X_test=np.concatenate([pgfl_test,graph_test],axis=1)

# ------------------------------------------------------------
# Normalization
# ------------------------------------------------------------

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# ------------------------------------------------------------
# Tensor Conversion
# ------------------------------------------------------------

device="cuda" if torch.cuda.is_available() else "cpu"

X_train=torch.tensor(X_train,dtype=torch.float32)
X_test=torch.tensor(X_test,dtype=torch.float32)

y_train=torch.tensor(y_train,dtype=torch.long)
y_test=torch.tensor(y_test,dtype=torch.long)

X_train=X_train.unsqueeze(1)
X_test=X_test.unsqueeze(1)

train_loader=DataLoader(
    TensorDataset(X_train,y_train),
    batch_size=32,
    shuffle=True
)

test_loader=DataLoader(
    TensorDataset(X_test,y_test),
    batch_size=32,
    shuffle=False
)

# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(self,d_model,max_len=5000):

        super().__init__()

        pe=torch.zeros(max_len,d_model)

        position=torch.arange(0,max_len).unsqueeze(1)

        div=torch.exp(
            torch.arange(0,d_model,2)
            *(-math.log(10000.0)/d_model)
        )

        pe[:,0::2]=torch.sin(position*div)
        pe[:,1::2]=torch.cos(position*div)

        self.pe=pe.unsqueeze(0)

    def forward(self,x):

        return x+self.pe[:,:x.size(1)].to(x.device)

# ============================================================
# Transformer Model
# ============================================================

class PGTSN(nn.Module):

    def __init__(self,input_dim,n_class):

        super().__init__()

        self.linear=nn.Linear(input_dim,256)

        self.pos=PositionalEncoding(256)

        encoder_layer=nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            dropout=0.2,
            batch_first=True
        )

        self.encoder=nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )

        self.attention=nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

        self.fc=nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,n_class)
        )

    def forward(self,x):

        x=self.linear(x)

        x=self.pos(x)

        x=self.encoder(x)

        w=torch.softmax(
            self.attention(x),
            dim=1
        )

        x=(x*w).sum(1)

        out=self.fc(x)

        return out,x

# ============================================================
# Model Initialization
# ============================================================

model=PGTSN(
    X_train.shape[2],
    len(np.unique(y_train.numpy()))
).to(device)

criterion=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.0005
)

# ============================================================
# Training
# ============================================================

print("Training PGTSN...")

for epoch in range(50):

    model.train()

    total_loss=0

    for x,y in train_loader:

        x=x.to(device)
        y=y.to(device)

        optimizer.zero_grad()

        pred,_=model(x)

        loss=criterion(pred,y)

        loss.backward()

        optimizer.step()

        total_loss+=loss.item()

    print(
        f"Epoch {epoch+1:02d}  Loss={total_loss/len(train_loader):.4f}"
    )

# ============================================================
# Generate Transformer Embeddings
# ============================================================

model.eval()

with torch.no_grad():

    _,train_embed=model(X_train.to(device))
    _,test_embed=model(X_test.to(device))

train_embed=train_embed.cpu().numpy()
test_embed=test_embed.cpu().numpy()

print(train_embed.shape)

# ============================================================
# Save Outputs
# ============================================================

pd.DataFrame(train_embed).to_csv(
    "Transformer_train.csv",
    index=False
)

pd.DataFrame(test_embed).to_csv(
    "Transformer_test.csv",
    index=False
)

torch.save(
    model.state_dict(),
    "PGTSN_Model.pth"
)

print("\nSTEP 5 COMPLETED SUCCESSFULLY")

# ============================================================
# STEP 6 : Multi-Scale Biological Feature Fusion
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# Load Features
# ------------------------------------------------------------

pgfl_train = pd.read_csv("PGFL_train.csv").values
pgfl_test  = pd.read_csv("PGFL_test.csv").values

graph_train = pd.read_csv("Graph_train.csv").values
graph_test  = pd.read_csv("Graph_test.csv").values

trans_train = pd.read_csv("Transformer_train.csv").values
trans_test  = pd.read_csv("Transformer_test.csv").values

# ------------------------------------------------------------
# Concatenate Features
# ------------------------------------------------------------

train_feature = np.concatenate(
    [pgfl_train, graph_train, trans_train],
    axis=1
)

test_feature = np.concatenate(
    [pgfl_test, graph_test, trans_test],
    axis=1
)

# ------------------------------------------------------------
# Normalize
# ------------------------------------------------------------

scaler = StandardScaler()

train_feature = scaler.fit_transform(train_feature)
test_feature = scaler.transform(test_feature)

train_tensor = torch.FloatTensor(train_feature).to(device)
test_tensor = torch.FloatTensor(test_feature).to(device)

# ============================================================
# Cross Attention Module
# ============================================================

class CrossAttention(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x):

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        score = torch.softmax(
            torch.matmul(q, k.T) / np.sqrt(x.size(1)),
            dim=1
        )

        out = torch.matmul(score, v)

        return out

# ============================================================
# Variational AutoEncoder
# ============================================================

class VAE(nn.Module):

    def __init__(self, input_dim, latent=128):

        super().__init__()

        self.fc1 = nn.Linear(input_dim,512)
        self.fc2 = nn.Linear(512,256)

        self.mu = nn.Linear(256,latent)
        self.logvar = nn.Linear(256,latent)

        self.d1 = nn.Linear(latent,256)
        self.d2 = nn.Linear(256,512)
        self.out = nn.Linear(512,input_dim)

    def encode(self,x):

        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))

        return self.mu(x),self.logvar(x)

    def reparameterize(self,mu,logvar):

        std=torch.exp(0.5*logvar)

        eps=torch.randn_like(std)

        return mu+eps*std

    def decode(self,z):

        z=torch.relu(self.d1(z))
        z=torch.relu(self.d2(z))

        return self.out(z)

    def forward(self,x):

        mu,logvar=self.encode(x)

        z=self.reparameterize(mu,logvar)

        recon=self.decode(z)

        return recon,mu,logvar,z

# ============================================================
# Initialize Network
# ============================================================

attention = CrossAttention(
    train_tensor.shape[1]
).to(device)

vae = VAE(
    train_tensor.shape[1],
    latent=128
).to(device)

optimizer = torch.optim.Adam(
    list(attention.parameters())+
    list(vae.parameters()),
    lr=0.0005
)

# ============================================================
# Train
# ============================================================

print("Training Cross-Attention + VAE...")

for epoch in range(50):

    optimizer.zero_grad()

    attention_feature = attention(train_tensor)

    recon,mu,logvar,z = vae(attention_feature)

    mse = ((recon-attention_feature)**2).mean()

    kl = -0.5*torch.mean(
        1+logvar-mu.pow(2)-logvar.exp()
    )

    loss = mse + 0.001*kl

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1:02d} Loss:{loss.item():.5f}"
    )

# ============================================================
# Extract Unified Representation
# ============================================================

with torch.no_grad():

    train_attention = attention(train_tensor)
    _,_,_,train_latent = vae(train_attention)

    test_attention = attention(test_tensor)
    _,_,_,test_latent = vae(test_attention)

train_latent = train_latent.cpu().numpy()
test_latent = test_latent.cpu().numpy()

print(train_latent.shape)

# ============================================================
# Save
# ============================================================

pd.DataFrame(train_latent).to_csv(
    "Fusion_train.csv",
    index=False
)

pd.DataFrame(test_latent).to_csv(
    "Fusion_test.csv",
    index=False
)

torch.save(
    vae.state_dict(),
    "Fusion_VAE.pth"
)

print("\nSTEP 6 COMPLETED SUCCESSFULLY")

# ============================================================
# STEP 7 : Quantum-Inspired Biomarker Optimization
# ============================================================

import numpy as np
import pandas as pd
import random

from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------
# Load Features
# ------------------------------------------------------------

X_train = pd.read_csv("Fusion_train.csv")
X_test = pd.read_csv("Fusion_test.csv")

y_train = pd.read_csv("y_train_step1.csv").values.ravel()

print("Input Shape :", X_train.shape)

# ============================================================
# mRMR (Mutual Information Ranking)
# ============================================================

print("\nRunning mRMR...")

mi = mutual_info_classif(
    X_train,
    y_train,
    random_state=42
)

rank = np.argsort(mi)[::-1]

top_features = rank[:80]

X_train_mrmr = X_train.iloc[:, top_features]
X_test_mrmr = X_test.iloc[:, top_features]

print("After mRMR :", X_train_mrmr.shape)

# ============================================================
# Bootstrap Stability Selection
# ============================================================

print("\nBootstrap Stability Selection...")

importance = np.zeros(X_train_mrmr.shape[1])

for i in range(100):

    idx = np.random.choice(
        len(X_train_mrmr),
        len(X_train_mrmr),
        replace=True
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=i
    )

    model.fit(
        X_train_mrmr.iloc[idx],
        y_train[idx]
    )

    importance += model.feature_importances_

importance /= 100

stable_index = np.argsort(importance)[::-1][:40]

X_train_boot = X_train_mrmr.iloc[:, stable_index]
X_test_boot = X_test_mrmr.iloc[:, stable_index]

print("Stable Biomarkers :", X_train_boot.shape)

# ============================================================
# Quantum-Behaved PSO
# ============================================================

print("\nRunning QPSO...")

dim = X_train_boot.shape[1]

particles = 30
iterations = 100

position = np.random.rand(particles, dim)

pbest = position.copy()

gbest = position[0].copy()

fitness = np.zeros(particles)

def evaluate(mask):

    mask = mask > 0.5

    if mask.sum() == 0:
        return 0

    feature = X_train_boot.iloc[:, mask]

    clf = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    clf.fit(feature, y_train)

    return clf.score(feature, y_train)

# Initial Fitness

for i in range(particles):

    fitness[i] = evaluate(position[i])

gbest = position[np.argmax(fitness)].copy()

# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------

beta = 0.75

for epoch in range(iterations):

    mbest = np.mean(pbest, axis=0)

    for i in range(particles):

        phi = np.random.rand(dim)

        p = phi * pbest[i] + (1 - phi) * gbest

        u = np.random.rand(dim)

        direction = np.where(
            np.random.rand(dim) > 0.5,
            1,
            -1
        )

        position[i] = p + direction * beta * np.abs(
            mbest - position[i]
        ) * np.log(1 / (u + 1e-10))

        position[i] = np.clip(position[i], 0, 1)

        score = evaluate(position[i])

        if score > fitness[i]:

            fitness[i] = score

            pbest[i] = position[i].copy()

    gbest = pbest[np.argmax(fitness)]

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch:03d}  Best Fitness = {fitness.max():.4f}"
        )

# ============================================================
# Selected Biomarkers
# ============================================================

mask = gbest > 0.5

X_train_final = X_train_boot.iloc[:, mask]
X_test_final = X_test_boot.iloc[:, mask]

print("\nOptimized Biomarker Shape :", X_train_final.shape)

# ============================================================
# Save
# ============================================================

X_train_final.to_csv(
    "Optimized_train.csv",
    index=False
)

X_test_final.to_csv(
    "Optimized_test.csv",
    index=False
)

print("\nSTEP 7 COMPLETED SUCCESSFULLY")

# ============================================================
# STEP 8 : Explainable Preventive Oncology Intelligence Module
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular

import torch
import torch.nn as nn

from captum.attr import IntegratedGradients

# ------------------------------------------------------------
# Load Optimized Features
# ------------------------------------------------------------

X_train = pd.read_csv("Optimized_train.csv")
X_test = pd.read_csv("Optimized_test.csv")

y_train = pd.read_csv("y_train_step1.csv").values.ravel()
y_test = pd.read_csv("y_test_step1.csv").values.ravel()

# ------------------------------------------------------------
# Convert Tensor
# ------------------------------------------------------------

device="cuda" if torch.cuda.is_available() else "cpu"

X_train_tensor=torch.FloatTensor(X_train.values).to(device)
X_test_tensor=torch.FloatTensor(X_test.values).to(device)

# ============================================================
# Risk Prediction Model
# ============================================================

class RiskModel(nn.Module):

    def __init__(self,input_dim,n_class):

        super().__init__()

        self.net=nn.Sequential(

            nn.Linear(input_dim,256),
            nn.ReLU(),

            nn.Linear(256,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,n_class)

        )

    def forward(self,x):

        return self.net(x)

model=RiskModel(
    X_train.shape[1],
    len(np.unique(y_train))
).to(device)

criterion=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# ============================================================
# Train
# ============================================================

print("Training Explainability Model...")

for epoch in range(50):

    optimizer.zero_grad()

    pred=model(X_train_tensor)

    loss=criterion(
        pred,
        torch.LongTensor(y_train).to(device)
    )

    loss.backward()

    optimizer.step()

    if epoch%10==0:

        print(epoch,loss.item())

# ============================================================
# SHAP
# ============================================================

print("\nGenerating SHAP Values...")

background=X_train.sample(100).values

explainer=shap.Explainer(
    model.cpu(),
    background
)

shap_values=explainer(
    X_test.iloc[:100]
)

shap.summary_plot(
    shap_values,
    X_test.iloc[:100],
    show=False
)

plt.savefig("SHAP_Summary.png",dpi=300)
plt.close()

# ============================================================
# LIME
# ============================================================

print("Generating LIME Explanation...")

explainer_lime=lime.lime_tabular.LimeTabularExplainer(

    training_data=X_train.values,

    feature_names=X_train.columns.tolist(),

    class_names=[
        str(i)
        for i in np.unique(y_train)
    ],

    mode="classification"

)

def predict_fn(x):

    x=torch.FloatTensor(x)

    with torch.no_grad():

        y=model(x)

        y=torch.softmax(y,1)

    return y.numpy()

exp=explainer_lime.explain_instance(

    X_test.iloc[0].values,

    predict_fn,

    num_features=10

)

fig=exp.as_pyplot_figure()

plt.savefig("LIME_Explanation.png",dpi=300)

plt.close()

# ============================================================
# Integrated Gradients
# ============================================================

print("Integrated Gradients...")

model=model.to(device)

ig=IntegratedGradients(model)

attr=ig.attribute(

    X_test_tensor[:50],

    target=0

)

importance=torch.mean(

    torch.abs(attr),

    dim=0

).cpu().numpy()

importance=pd.DataFrame({

    "Feature":X_train.columns,

    "Importance":importance

})

importance=importance.sort_values(

    by="Importance",

    ascending=False

)

importance.to_csv(

    "IntegratedGradients.csv",

    index=False

)

# ============================================================
# Top Biomarkers
# ============================================================

top20=importance.head(20)

plt.figure(figsize=(8,6))

plt.barh(

    top20["Feature"].astype(str),

    top20["Importance"]

)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(

    "Top20_Biomarkers.png",

    dpi=300

)

plt.close()

print("\nExplainability Reports Generated")

print("SHAP_Summary.png")

print("LIME_Explanation.png")

print("IntegratedGradients.csv")

print("Top20_Biomarkers.png")

print("\nSTEP 8 COMPLETED SUCCESSFULLY")


# ============================================================
# STEP 9 : Hybrid Deep Risk Prediction Network
# STEP 10: Personalized Preventive Oncology Recommendation
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from torch.utils.data import TensorDataset,DataLoader

device="cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------

X_train=pd.read_csv("Optimized_train.csv").values
X_test=pd.read_csv("Optimized_test.csv").values

y_train=pd.read_csv("y_train_step1.csv").values.ravel()
y_test=pd.read_csv("y_test_step1.csv").values.ravel()

X_train=torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)

y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)

train_loader=DataLoader(
    TensorDataset(X_train,y_train),
    batch_size=32,
    shuffle=True
)

test_loader=DataLoader(
    TensorDataset(X_test,y_test),
    batch_size=32,
    shuffle=False
)

# ============================================================
# Residual Block
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self,dim):

        super().__init__()

        self.block=nn.Sequential(

            nn.Linear(dim,dim),
            nn.ReLU(),

            nn.Linear(dim,dim)

        )

    def forward(self,x):

        return torch.relu(x+self.block(x))

# ============================================================
# Hybrid Deep Risk Network
# ============================================================

class HybridRiskNet(nn.Module):

    def __init__(self,input_dim,n_class):

        super().__init__()

        self.input=nn.Linear(input_dim,256)

        self.res1=ResidualBlock(256)
        self.res2=ResidualBlock(256)

        self.attention=nn.Sequential(

            nn.Linear(256,128),
            nn.Tanh(),
            nn.Linear(128,1)

        )

        self.classifier=nn.Sequential(

            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,n_class)

        )

    def forward(self,x):

        x=torch.relu(self.input(x))

        x=self.res1(x)
        x=self.res2(x)

        w=torch.softmax(
            self.attention(x),
            dim=0
        )

        x=x*w

        out=self.classifier(x)

        return out

# ============================================================
# Initialize
# ============================================================

model=HybridRiskNet(
    X_train.shape[1],
    len(torch.unique(y_train))
).to(device)

criterion=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.0005
)

# ============================================================
# Train
# ============================================================

print("Training Hybrid Risk Network...")

for epoch in range(100):

    model.train()

    total_loss=0

    for x,y in train_loader:

        x=x.to(device)
        y=y.to(device)

        optimizer.zero_grad()

        pred=model(x)

        loss=criterion(pred,y)

        loss.backward()

        optimizer.step()

        total_loss+=loss.item()

    if epoch%10==0:

        print(
            f"Epoch {epoch:03d} Loss:{total_loss/len(train_loader):.4f}"
        )

torch.save(
    model.state_dict(),
    "HybridRiskNet.pth"
)

# ============================================================
# Prediction
# ============================================================

model.eval()

pred=[]
prob=[]

with torch.no_grad():

    for x,y in test_loader:

        x=x.to(device)

        out=model(x)

        p=torch.softmax(out,1)

        pred.extend(
            torch.argmax(
                p,
                1
            ).cpu().numpy()
        )

        prob.extend(
            p.cpu().numpy()
        )

pred=np.array(pred)
prob=np.array(prob)

# ============================================================
# Metrics
# ============================================================

acc=accuracy_score(y_test,pred)
pre=precision_score(y_test,pred,average="weighted")
rec=recall_score(y_test,pred,average="weighted")
f1=f1_score(y_test,pred,average="weighted")

print("\n==========================")
print("Accuracy :",acc)
print("Precision:",pre)
print("Recall   :",rec)
print("F1 Score :",f1)
print("==========================")

# ============================================================
# Confusion Matrix
# ============================================================

cm=confusion_matrix(y_test,pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()

plt.savefig(
    "ConfusionMatrix.png",
    dpi=300
)

plt.close()

# ============================================================
# ROC Curve
# ============================================================

if len(np.unique(y_test))==2:

    fpr,tpr,_=roc_curve(
        y_test,
        prob[:,1]
    )

    roc_auc=auc(fpr,tpr)

    plt.figure(figsize=(6,5))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC={roc_auc:.4f}"
    )

    plt.plot([0,1],[0,1],'--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "ROC_Curve.png",
        dpi=300
    )

    plt.close()

# ============================================================
# Classification Report
# ============================================================

report=classification_report(
    y_test,
    pred
)

with open(
    "Classification_Report.txt",
    "w"
) as f:

    f.write(report)

# ============================================================
# Personalized Risk Report
# ============================================================

risk_names=[
    "Low Risk",
    "Medium Risk",
    "High Risk",
    "Very High Risk"
]

results=[]

for i,p in enumerate(pred):

    if p>=len(risk_names):
        label=risk_names[-1]
    else:
        label=risk_names[p]

    results.append([

        i+1,

        label,

        float(np.max(prob[i]))

    ])

risk_df=pd.DataFrame(

    results,

    columns=[

        "Patient_ID",

        "Risk_Level",

        "Probability"

    ]

)

risk_df.to_csv(

    "Cancer_Risk_Report.csv",

    index=False

)

print("\nGenerated Files")

print("HybridRiskNet.pth")
print("ConfusionMatrix.png")
print("ROC_Curve.png")
print("Classification_Report.txt")
print("Cancer_Risk_Report.csv")

print("\nPROJECT COMPLETED SUCCESSFULLY")