# 📚 Ringkasan *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*

Repositori ini berisi rangkuman catatan dan latihan praktis dari buku *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. Buku ini memberikan fondasi kuat serta praktik nyata untuk membangun dan mengimplementasikan model Machine Learning modern.

---

## 📖 Ringkasan Per Chapter

### **Chapter 1 — The Machine Learning Landscape**

- Menjelaskan apa itu Machine Learning dan mengapa penting.
- Perbedaan supervised, unsupervised, dan reinforcement learning.
- Contoh kasus nyata seperti spam detection, rekomendasi, dan self-driving.
- Memperkenalkan dataset, fitur, label, dan pipeline dasar ML.

---

### **Chapter 2 — End-to-End Machine Learning Project**

- Panduan membangun proyek ML secara end-to-end.
  - Eksplorasi data (EDA).
  - Membersihkan dan menyiapkan data.
  - Feature engineering.
  - Splitting data: training, validation, test.
  - Cross-validation & hyperparameter tuning.
- Contoh: Prediksi harga rumah di California.

---

### **Chapter 3 — Classification**

- Fokus pada masalah klasifikasi (binary & multiclass).
- Algoritma: Logistic Regression, K-Nearest Neighbors.
- Cara mengukur performa: precision, recall, F1-score.
- Penggunaan confusion matrix & ROC curve.
- Studi kasus: klasifikasi angka tangan MNIST.

---

### **Chapter 4 — Training Models**

- Konsep regresi linear dan gradient descent.
- Cara kerja batch, mini-batch, dan stochastic gradient descent.
- Tips untuk convergence dan mencegah stuck di local minima.
- Visualisasi error dan loss function.

---

### **Chapter 5 — Support Vector Machines**

- Penjelasan margin optimal untuk klasifikasi.
- Soft margin vs hard margin.
- Kernel trick untuk data non-linear.
- SVM untuk regresi dan klasifikasi.
- Visualisasi decision boundary & support vectors.

---

### **Chapter 6 — Decision Trees**

- Struktur pohon keputusan: root, node, leaf.
- Algoritma splitting dan impurity.
- Overfitting pada tree yang terlalu dalam.
- Pruning pohon untuk generalisasi lebih baik.
- Kelebihan interpretabilitas Decision Trees.

---

### **Chapter 7 — Ensemble Learning and Random Forests**

- Konsep ensemble learning: menggabungkan beberapa model lemah menjadi model kuat.
- Metode:
  - Bagging & Pasting.
  - Random Forest.
  - Boosting (AdaBoost, Gradient Boosting).
  - Stacking.
- Perbandingan akurasi ensemble vs model tunggal.

---

### **Chapter 8 — Dimensionality Reduction**

- Reduksi dimensi untuk menangani data berdimensi tinggi.
- Metode:
  - Principal Component Analysis (PCA).
  - Kernel PCA.
  - Manifold Learning (LLE, t-SNE).
- Visualisasi data hasil reduksi ke 2D.

---

### **Chapter 9 — Unsupervised Learning**

- Clustering:
  - K-Means Clustering.
  - DBSCAN.
  - Gaussian Mixture Model.
- Association Rules:
  - Mencari pola item yang sering muncul bersamaan.
- Contoh: Segmentasi customer & Market Basket Analysis.

---

### **Chapter 10 — Introduction to Artificial Neural Networks**

- Dasar Neural Networks:
  - Perceptron.
  - Activation Function (ReLU, Sigmoid).
  - Backpropagation.
- Implementasi jaringan saraf sederhana.
- Penjelasan multilayer perceptron (MLP).

---

### **Chapter 11 — Training Deep Neural Networks**

- Teknik stabil training DNN:
  - Weight Initialization (He, Xavier).
  - Batch Normalization.
  - Dropout untuk regularisasi.
  - Early Stopping.
  - Optimizer modern: Adam, RMSProp.
- Learning rate scheduling & gradient clipping.

---

### **Chapter 12 — Custom Models and Training with TensorFlow**

- Cara membangun:
  - Layer kustom.
  - Loss function manual.
  - Training loop kustom dengan `GradientTape`.
- Contoh: Implementasi layer dan training step sendiri.

---

### **Chapter 13 — Loading and Preprocessing Data with TensorFlow**

- Membuat pipeline data dengan `tf.data`:
  - Membaca file besar.
  - Shuffle, batch, prefetch.
  - Auto-tuning pipeline.
- Penggunaan format TFRecord untuk performa tinggi.
- Integrasi dengan TensorFlow Datasets (TFDS).

---

### **Chapter 14 — Deploying TensorFlow Models to Production**

- Menyimpan model dengan format SavedModel.
- Serving model:
  - TensorFlow Serving.
  - REST API & gRPC.
- Versioning model.
- Monitoring & logging.
- Batching request untuk throughput tinggi.
- Strategi A/B testing & canary deployment.

---

### **Chapter 15 — Distributing TensorFlow Across Devices and Servers**

- Strategi distribusi:
  - MirroredStrategy (multi-GPU satu mesin).
  - MultiWorkerMirroredStrategy (multi-node).
  - TPUStrategy (untuk TPU di cloud).
- Sinkronisasi gradien antar replica.
- Sharding dataset otomatis.
- Tips penggunaan pipeline data terdistribusi.

---

### **Chapter 16 — Natural Language Processing with RNNs and Attention**

- NLP workflow:
  - Tokenization & Padding.
  - Word Embedding.
  - RNN, LSTM, GRU.
  - Bidirectional RNN.
  - Sequence-to-Sequence (Seq2Seq).
  - Attention Mechanism.
- Pengantar Transformer dan model NLP modern.

---

### **Chapter 17 — Representing and Reasoning with Knowledge**

- Konsep Symbolic AI:
  - Propositional Logic.
  - First Order Logic.
  - Resolution & inference.
- Knowledge Base (KB) & rule-based systems.
- Probabilistic Reasoning (Bayesian Network, Markov Networks).
- Neuro-Symbolic Systems: kombinasi ML & reasoning.

---

### **Chapter 18 — Reinforcement Learning**

- Konsep RL:
  - Agent, Environment, State, Action, Reward.
  - Policy & Value Function.
- Markov Decision Process (MDP) & Bellman Equation.
- Algoritma:
  - Q-Learning.
  - SARSA.
  - Deep Q-Network (DQN).
- Exploration vs Exploitation (ε-greedy).
- Contoh Q-Learning tabular.

---

### **Chapter 19 — Training and Deploying TensorFlow Models at Scale**

- Training skala besar:
  - Sinkronisasi gradien.
  - Multi-worker training.
  - Distribusi di cluster & cloud.
- Pipeline data besar.
- Serving model di production.
- Monitoring performa, A/B testing, canary deployment.

---

## ✅ Kesimpulan

Buku ini membimbing pembaca mulai dari teori dasar hingga penerapan Machine Learning dan Deep Learning skala industri. Praktik coding, tips engineer, dan best practice menjadikan materi relevan untuk developer, data scientist, peneliti, hingga engineer ML production.

---

## 📌 Catatan

Konten ini adalah rangkuman pembelajaran pribadi, **bukan pengganti buku asli**. Disarankan membaca buku resminya untuk detail mendalam.

---

Terima kasih telah berkunjung! 🚀✨
