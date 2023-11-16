import torch
import torchvision.models as models
import torch.nn as nn
from const import *

# Charger un modèle pré-entrainé
model = models.resnet18(pretrained=True)


model.fc = nn.Linear(model.fc.in_features, NB_CLASSES)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 10

train_loader = []
val_loader = []


for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:  # Utilisez votre propre DataLoader
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Évaluation sur l'ensemble de validation (facultatif)

model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:  # Utilisez votre propre DataLoader
        outputs = model(inputs)
        # Calculer les métriques d'évaluation (précision, rappel, etc.)

torch.save(model.state_dict(), 'modele_entrene.pth')

