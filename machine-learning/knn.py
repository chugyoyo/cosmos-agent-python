# 1. 导入必要的库
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. 加载数据（鸢尾花数据集）
iris = datasets.load_iris()
X = iris.data  # 特征：萼片长度、萼片宽度、花瓣长度、花瓣宽度
y = iris.target  # 标签：0=Setosa, 1=Versicolor, 2=Virginica

print("数据集信息:")
print(f"特征形状: {X.shape}")  # 150个样本，4个特征
print(f"标签形状: {y.shape}")
print(f"类别名称: {iris.target_names}")
print(f"前5个样本的特征:\n{X[:5]}")
print(f"前5个样本的标签: {y[:5]}")

# 3. 数据预处理
# 拆分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 标准化特征（KNN对特征尺度敏感，必须标准化！）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n训练集大小: {X_train_scaled.shape[0]} 个样本")
print(f"测试集大小: {X_test_scaled.shape[0]} 个样本")

# 4. 创建并训练KNN模型
# 注意：KNN的"训练"实际上只是存储数据，没有复杂的计算
k = 3  # 选择K=3
knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn.fit(X_train_scaled, y_train)  # 这里只是存储标准化后的训练数据

print(f"\n使用 K={k} 的KNN模型")
print("模型参数:", knn.get_params())

# 5. 进行预测
y_pred = knn.predict(X_test_scaled)

# 6. 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"\n模型准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 7. 手动演示KNN的工作原理（对单个新样本）
print("\n" + "="*60)
print("手动演示KNN如何工作:")
print("="*60)

# 创建一个虚拟的新样本（类似一朵新花的特征）
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # 类似Setosa花的特征
new_sample_scaled = scaler.transform(new_sample)

# 查看这个样本的K个最近邻
distances, indices = knn.kneighbors(new_sample_scaled)
print(f"\n新样本: {new_sample[0]}")
print(f"标准化后的新样本: {new_sample_scaled[0]}")

print(f"\n最近的 {k} 个邻居的索引: {indices[0]}")
print(f"这些邻居的距离: {distances[0]}")

print(f"\n这些邻居的标签:")
for i, idx in enumerate(indices[0]):
    original_label = y_train[idx]
    label_name = iris.target_names[original_label]
    print(f"  邻居 {i+1}: 训练样本 #{idx}, 标签='{label_name}' (距离={distances[0][i]:.4f})")

# 进行预测
prediction = knn.predict(new_sample_scaled)
predicted_label = iris.target_names[prediction[0]]
print(f"\n预测结果: 这朵花是 '{predicted_label}'")

# 8. 查看预测的概率（软分类）
probabilities = knn.predict_proba(new_sample_scaled)
print(f"\n属于各个类别的概率:")
for i, class_name in enumerate(iris.target_names):
    print(f"  {class_name}: {probabilities[0][i]:.4f}")

# 9. 尝试不同的K值，看看效果如何
print("\n" + "="*60)
print("尝试不同的K值:")
print("="*60)

k_values = [1, 3, 5, 7, 10, 15]
for k_val in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k_val)
    knn_temp.fit(X_train_scaled, y_train)
    y_pred_temp = knn_temp.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred_temp)
    print(f"K={k_val:2d}: 准确率 = {acc:.4f}")