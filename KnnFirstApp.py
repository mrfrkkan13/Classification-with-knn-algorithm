from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y): #It gives the model the training data and labels required for the KNN algorithm.
        self.X_train = X
        self.y_train = y

    def predict(self, X): #makes predictions for the input data and returns these predictions as an array.
        y_pred = [self._predict(x) for x in X] 
        return np.array(y_pred)

    def _predict(self, x): #includes the KNN algorithm used to make predictions for input features.
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Iris veri setini yükleme
iris = load_iris()
X = iris.data
y = iris.target

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi standartlaştırma
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN modelini oluşturma ve eğitme
knn = KNN(k=3)
knn.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred = knn.predict(X_test)

# Doğruluk değerini hesaplama
accuracy = np.mean(y_pred == y_test)
print("Doğruluk:", accuracy)
print(iris.target_names)

# Doğruluk sonucunu ve tahmin edilen sınıfları görselleştirme
correct_count = np.sum(y_pred == y_test)
total_count = len(y_test)
incorrect_count = total_count - correct_count

plt.figure(figsize=(8, 6))
plt.bar(['Doğru', 'Yanlış'], [correct_count, incorrect_count], color=['green', 'red'])
plt.xlabel('Sonuç')
plt.ylabel('Sayı')
plt.title('Doğruluk Sonucu')
plt.ylim(0, total_count)
plt.text(0, correct_count, f"{correct_count} ({accuracy*100:.2f}%)", ha='center', va='bottom')
plt.text(1, incorrect_count, f"{incorrect_count} ({(1-accuracy)*100:.2f}%)", ha='center', va='bottom')
plt.show()

# Tahmin edilen ve gerçek sınıfları karşılaştır
plt.figure(figsize=(10, 6))

# Gerçek sınıfları göster
plt.bar(np.arange(len(y_test))-0.2, y_test, width=0.4, alpha=0.5, label='Gerçek Sınıf')

# Tahmin edilen sınıfları göster
plt.bar(np.arange(len(y_pred))+0.2, y_pred, width=0.4, alpha=0.5, label='Tahmin Edilen Sınıf')

plt.xlabel('Örnekler')
plt.ylabel('Sınıflar')
plt.title('Gerçek ve Tahmin Edilen Sınıflar')
plt.legend()

# X eksenindeki etiketleri ayarla
plt.xticks(np.arange(len(y_test)), np.arange(len(y_test))+1)

plt.show()

# Karmaşıklık matrisini oluşturma
cm = confusion_matrix(y_test,y_pred)

# Karmaşıklık matrisini görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Karmaşıklık Matrisi')
plt.show()
