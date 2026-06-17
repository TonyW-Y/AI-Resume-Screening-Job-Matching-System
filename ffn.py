import torch
import torch.nn as nn

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader



class ffn(nn.Module):
    def __init__(self, in_features=768, h1=256, h2=128, out_features=24, drop=0.2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(in_features,h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Dropout(drop),

            nn.Linear(h2,out_features)
        )
    def forward(self, x):
        x = self.network(x)
        return x
    
def main():
    # load data
    df = pd.read_csv("data/master_resumes.csv")
    embeddings = np.load("embeddings/resumes.npy")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Set manual seed
    torch.manual_seed(42)

    # turns the category text into recognizable numbers
    le = LabelEncoder()
    labels = le.fit_transform(df["category"])

    # convert to tensor
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    # split training and testing data (X=embeddings, y=labels)
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    # create DataLoaders so we process the data in small batches
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=32)

    # move data from CPU to MPS
    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)

    # Initialize model
    model = ffn().to(device)

    # Initialize Loss function
    loss_function = nn.CrossEntropyLoss()

    # Initialize Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    epochs = 30

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            # move data from CPU to MPS
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            # reset gradient to zero
            optimizer.zero_grad()
            # calculate prediction
            prediction = model(X_batch)
            # find the loss between prediction and actual answer
            loss = loss_function(prediction, y_batch)
            # look back and see which gradiants made the mistake
            loss.backward()
            # update the gradiants
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    # evaluate the model
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # save the model
    torch.save(model.state_dict(), "models/classifier_model.pth")
    print("✅ Model saved to models/classifier_model.pth")

if __name__ == "__main__":
    main()
