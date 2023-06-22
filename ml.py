import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

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

# create and train the model with training data
model = GradientBoostingClassifier() # we could select several classifier, this one is the best I tested
model.fit(X_train, y_train)

# predict on test data with trained model
y_pred = model.predict(X_test)

# calculate the acuracy
accuracy = accuracy_score(y_test, y_pred)
print("accuracy：", accuracy)


print("finished")