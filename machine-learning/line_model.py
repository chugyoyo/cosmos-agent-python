import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- 1. 创建模拟数据集 ---
# 假设有 5 组数据 (m=5)
data = {
    'Area_sqft': [1500, 2000, 1200, 3500, 1800],
    'Rooms':     [3, 4, 2, 5, 3],
    'Price_k':   [300, 450, 250, 700, 350] # 价格 (千美元)
}
df = pd.DataFrame(data)
print("df = " + str(df))

# 特征矩阵 X (需要添加一列1作为截距项 x0)
# X_original: m x n (n=2, 面积和房间数)
X_original = df[['Area_sqft', 'Rooms']].values
print("X_original = " + str(X_original))

# 目标向量 y
y = df['Price_k'].values
print("y = " + str(y))

# 添加偏置项 x0=1 到特征矩阵 X
# X: m x (n+1)
X = np.hstack([np.ones((len(X_original), 1)), X_original])


# --- 2. 正规方程求解 ---
# 计算 (X转置 * X)
X_T_X = X.T @ X
print("X_T_X = " + str(X_T_X))

# 计算 (X转置 * X) 的逆
# 使用 np.linalg.inv() 进行矩阵求逆
try:
    X_T_X_inv = np.linalg.inv(X_T_X)
except np.linalg.LinAlgError:
    print("矩阵不可逆，无法使用正规方程。")
    exit()

# 计算 (X转置 * y)
X_T_y = X.T @ y
print("X_T_y = " + str(X_T_y))

# 最终求解参数 theta (包含截距b和权重w1, w2)
# 对应公式：$$\mathbf{\theta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
theta_normal_equation = X_T_X_inv @ X_T_y
print("theta_normal_equation = " + str(theta_normal_equation))

# 提取参数
b_ne = theta_normal_equation[0]
w1_ne = theta_normal_equation[1]
w2_ne = theta_normal_equation[2]

print("## 📊 方法一：正规方程求解结果 ##")
print(f"参数 theta (b, w1, w2): {theta_normal_equation}")
print(f"截距 b (基础价格): {b_ne:.4f} 千美元")
print(f"面积权重 w1: {w1_ne:.4f} (每平方英尺价格提升)")
print(f"房间数权重 w2: {w2_ne:.4f} (每房间价格提升)\n")

# --- 3. Scikit-learn 库求解 ---
# 创建线性回归模型实例
model = LinearRegression()

# 训练模型 (拟合数据)
# 注意：Scikit-learn 的 fit 函数会自动处理截距项，所以我们使用 X_original
model.fit(X_original, y)
print("model.coef_ = " + str(model.coef_))

# 提取参数
b_skl = model.intercept_
w_skl = model.coef_

print("## ⚙️ 方法二：Scikit-learn 库求解结果 ##")
print(f"截距 b (基础价格): {b_skl:.4f} 千美元")
print(f"权重 w (w1, w2): {w_skl}")
print(f"面积权重 w1: {w_skl[0]:.4f}")
print(f"房间数权重 w2: {w_skl[1]:.4f}\n")

# --- 4. 评估与预测 ---

# 预测结果
y_pred_ne = X @ theta_normal_equation
y_pred_skl = model.predict(X_original)

# 计算均方误差 (MSE)
mse_ne = mean_squared_error(y, y_pred_ne)
mse_skl = mean_squared_error(y, y_pred_skl)

print("## 🎯 模型评估 ##")
print(f"正规方程 MSE: {mse_ne:.4f}")
print(f"Scikit-learn MSE: {mse_skl:.4f}")

# 🚀 应用：预测一套新房子的价格
new_house = np.array([[2200, 4]]) # 2200平方英尺，4个房间

# 正规方程预测 (手动添加截距 x0=1)
new_house_ne = np.hstack([1, new_house[0]])
price_ne = new_house_ne @ theta_normal_equation

# Scikit-learn 预测
price_skl = model.predict(new_house)

print("\n## 💰 新房价格预测 (2200平方英尺, 4房间) ##")
print(f"正规方程预测价格: {price_ne:.2f} 千美元")
print(f"Scikit-learn预测价格: {price_skl[0]:.2f} 千美元")

# 验证两种方法的结果是否一致
print(f"\n两种方法参数是否接近: {np.allclose(theta_normal_equation, np.insert(w_skl, 0, b_skl))}")