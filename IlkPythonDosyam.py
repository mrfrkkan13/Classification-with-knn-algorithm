import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import KnnFirstApp

# Iris veri setini yükleme ve KNN modelini eğitme işlemleri
# (Bu kısmı kodunuzun üzerine yazmayın, sadece görselleştirme kısmını ekleyin)

# Karmaşıklık matrisini oluşturma
cm = confusion_matrix(KnnFirstApp.y_test, KnnFirstApp.y_pred)

# Karmaşıklık matrisini görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Karmaşıklık Matrisi')
plt.show()
