import pandas as pd
import torch
import torch.nn as nn
#import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch_optimizer as optim

from torch.utils.tensorboard import SummaryWriter   

# read the excel file
data = pd.read_excel("MealAnalysis(2017).xlsx")
# drop C to J in excel 
data = data.drop(columns=["gender", "age", "height", "weight", "EER[kcal]", "P target(15%)[g]", "F target(25%)[g]", "C target(60%)[g]"])

# extract features and the target
features = data.iloc[:, 1:17]  # Select colomns from B to Q as input feature(if above colomns not droped)
target = data["Score(1:worst 2:bad 3:good 4:best)"]  # select the colomn R as the target

# processing those non-value type with One-Hot and concat with other features
categorical_features = features.select_dtypes(include=["object"]).columns
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(features[categorical_features])
feature_names = encoder.get_feature_names_out(categorical_features)
encoded_features = pd.DataFrame(encoded_features, columns=feature_names)
features = pd.concat([features.drop(categorical_features, axis=1), encoded_features], axis=1)

# fill the N/A with mean value
features.fillna(features.mean(), inplace=True)

# feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# split train and test dataset randomly
X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)

# transform to pytorch tensor, for the target/y, we need to minus 1 to start with 0
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)-1
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long) - 1

# define the neural network, here I define 4 fc layers with softmax output
# as the dataset is small, we don't need to use heavy NN, otherwise will 
# easy to overfiting
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 4)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(x)
        x= F.log_softmax(x, dim=1)
        return x

# instantiate the model
model = NeuralNetwork(input_size=X_train_tensor.shape[1])

# define the loss function and optimizer, as well as the tensorboard writer
# to record the training/evaluation
criterion = nn.CrossEntropyLoss()
optimizer = optim.Ranger(model.parameters(), lr=0.001) 
writer = SummaryWriter()

# train the model
epochs = 709 # epochs number could be changed, please refer to my slides on why I select this value
for epoch in range(epochs):
    # Forward
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # BP and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    writer.add_scalar('training loss', loss, epoch)

    # print training loss and evaluate the accuracy
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = accuracy_score(y_test_tensor, predicted)
            writer.add_scalar('accuracy', accuracy, epoch)

# final evaluation on the test data
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)

# calculate the accuracy
accuracy = accuracy_score(y_test_tensor, predicted)
print("accuracy：", accuracy)

print("finished!")
