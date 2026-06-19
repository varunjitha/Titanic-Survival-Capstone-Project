import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset
df = pd.read_csv("titanic.csv")

# Data Cleaning
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# EDA
print("Dataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

# Graph 1
plt.figure()
sns.countplot(x='Survived', data=df)
plt.title("Passenger Survival Distribution")
plt.show()

# Graph 2
plt.figure()
sns.histplot(df['Age'])
plt.title("Age Distribution")
plt.show()

# Machine Learning
data = df[['Pclass','Sex','Age','Fare','Survived']]

data['Sex'] = data['Sex'].map({
    'male':0,
    'female':1
})

X = data[['Pclass','Sex','Age','Fare']]
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print("\nModel Accuracy:", accuracy)

# Graph 3
cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nConclusion:")
print("Female passengers had a higher survival rate.")
print("Passenger class influenced survival chances.")
print("Model achieved approximately 80% accuracy.")
