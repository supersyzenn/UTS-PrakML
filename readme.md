# README - Decision Tree Classification: Dataset Buys Comp UTS Praktikum ML

**Nama:** Ahmad Juaeni Yunus
**NIM:** 1227050011
**Praktikum:** Pembelajaran Mesin - E


## 1. Import Library

Langkah pertama, kita butuh beberapa library penting untuk proses pembacaan data, modeling, evaluasi, dan visualisasi.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
```

---

## 2. Baca dan Cek Datanya

Dataset diupload langsung lewat Google Colab. Setelah itu kita tampilkan beberapa baris awal untuk tahu isi dan struktur datanya.

```python
from google.colab import files
import pandas as pd

# Upload file
uploaded = files.upload()

# Ambil nama file yang diupload
filename = list(uploaded.keys())[0]

# Membaca dataset
df = pd.read_csv(filename)

# Menampilkan 5 baris pertama dari dataset
print(df.head())
print(df.info())
```

---

## 3. Preprocessing (Label Encoding & Split Data)
Sebelum dipakai untuk training model, data perlu diubah dulu. Kolom-kolom yang masih bertipe objek (string) diubah jadi angka pakai LabelEncoder. Lalu, kita pisahkan antara fitur dan label.

# Encode kolom kategorikal ke angka (kalau ada string)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# isahkan fitur dan target
# Ubah sesuai nama kolom target di dataset kamu
target_column = 'Buys_Computer'  # Ganti kalau berbeda
X = df.drop(columns=[target_column])
y = df[target_column]

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## 4. Buat dan Latih Model Decision Tree

Nah selajutnya kita panggil Decision Tree dan latih dengan data training yang udah kita siapkan.

```python
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

---

## 5. Prediksi dan Evaluasi Model

Model digunakan untuk memprediksi data uji. Setelah itu kita hitung akurasi, tampilkan confusion matrix dan classification report.

```python
y_pred = model.predict(X_test)

print("\nAkurasi:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```
---

## 6. Visualisasi Confusion Matrix (Heatmap)
Biar lebih enak dilihat, confusion matrix ditampilkan dalam bentuk heatmap.

```python
cm = confusion_matrix(y_test, y_pred)
classes = le.classes_ if hasattr(le, "classes_") else ["0", "1"]

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.show()
```
---

## 7. Visualisasi Distribusi Prediksi
Distribusi prediksi terhadap label asli juga divisualisasikan agar kita tahu bagaimana model memetakan hasil prediksinya.

```python
# Membuat DataFrame hasil prediksi
results_df = pd.DataFrame({'True Label': y_test, 'Predicted Label': y_pred})

# Mapping label numerik ke label asli (misalnya 'yes' dan 'no')
# Cek dulu kelas yang digunakan
label_map = {i: label for i, label in enumerate(le.classes_)}
results_df['True Label'] = results_df['True Label'].map(label_map)
results_df['Predicted Label'] = results_df['Predicted Label'].map(label_map)

# Membuat plot distribusi prediksi
plt.figure(figsize=(10, 6))
sns.countplot(data=results_df, x='True Label', hue='Predicted Label', palette='Set2')

# Tambahkan label dan judul
plt.xlabel('True Label')
plt.ylabel('Count')
plt.title('Distribusi Prediksi Berdasarkan Label Asli')
plt.legend(title='Predicted Label')
plt.show()
```
---

## 8. Kesimpulan & Evaluasi
Dari hasil di atas kita bisa menyimpulkan:

- Akurasi menunjukkan seberapa sering model memprediksi dengan benar.
- Confusion Matrix memberi gambaran detail antara label asli dan prediksi.
- Classification Report memberi metrik precision, recall, dan f1-score.
- Visualisasi bantu memahami performa model secara intuitif.

---

## Sumber Data

Dataset yang dipakai: `dataset_buys_comp.csv` 


