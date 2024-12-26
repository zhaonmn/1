# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB  # 导入朴素贝叶斯分类器
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import ydata_profiling as pp  # 替换为 ydata_profiling
from matplotlib import rcParams

# 设置字体为 SimHei（黑体），以支持中文
rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
rcParams['axes.unicode_minus'] = False  # 正常显示负号


# 1. 数据集加载与介绍
def load_and_describe_data():
    print("Iris 鸢尾花数据集介绍：")
    print("Iris 数据集是一个经典数据集，包含 3 类共 150 条记录，每类各 50 条数据。")
    print("每条数据有 4 项特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度。")
    print("目标是预测鸢尾花属于以下 3 种类别之一：")
    print("- Iris-setosa (0)")
    print("- Iris-versicolour (1)")
    print("- Iris-virginica (2)")
    print()

    # 加载数据集
    iris_dataset = load_iris()
    X = pd.DataFrame(iris_dataset['data'], columns=iris_dataset['feature_names'])
    y = pd.DataFrame(iris_dataset['target'], columns=['species'])
    iris_data = pd.concat([X, y], axis=1)

    # 数据预览
    print("Iris 数据集特征 (前 5 条记录)：")
    print(X.head())
    print("\nIris 数据集标签 (前 5 条记录)：")
    print(y.head())

    return X, y, iris_data, iris_dataset


# 2. 数据探索与可视化
def explore_and_visualize_data(X, y, iris_data):
    # 数据统计信息
    print("\n数据集统计信息：")
    print(iris_data.describe())

    # 箱线图
    iris_data.iloc[:, :-1].plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False, figsize=(10, 8))
    plt.suptitle("箱线图")
    plt.tight_layout()
    plt.show()

    # 直方图
    iris_data.iloc[:, :-1].hist(figsize=(10, 8))
    plt.suptitle("直方图")
    plt.tight_layout()
    plt.show()

    # 特征关系矩阵图
    sns.pairplot(data=iris_data, hue='species', diag_kind='kde', palette='Set2')
    plt.suptitle("特征关系矩阵图", y=1.02)
    plt.show()

    # 热图
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
    plt.title("特征相关性热图")
    plt.show()


# 3. 模型训练与评估
def train_and_evaluate_models(X_train, X_test, y_train, y_test, iris_dataset):
    # 模型字典，包含所有要训练的分类器
    models = {
        "K-近邻 (KNN)": KNeighborsClassifier(n_neighbors=3),
        "逻辑回归 (Logistic Regression)": LogisticRegression(max_iter=200),
        "支持向量机 (SVM)": SVC(),
        "决策树 (Decision Tree)": DecisionTreeClassifier(),
        "朴素贝叶斯 (Gaussian Naive Bayes)": GaussianNB()  # 添加朴素贝叶斯分类器
    }

    print("\n各分类器性能评估：")
    for name, model in models.items():
        # 训练模型
        model.fit(X_train, y_train.values.ravel())
        # 预测
        y_pred = model.predict(X_test)
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} 准确率：{accuracy:.3f}")

        # 打印分类报告
        print(f"\n{name} 分类报告：")
        print(classification_report(y_test, y_pred, target_names=iris_dataset.target_names))

        # 绘制混淆矩阵
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
                    xticklabels=iris_dataset.target_names,
                    yticklabels=iris_dataset.target_names)
        plt.title(f"{name} 混淆矩阵")
        plt.xlabel("预测值")
        plt.ylabel("真实值")
        plt.show()


# 4. 超参数调优示例
def hyperparameter_tuning(X_train, y_train):
    print("\n超参数调优示例：支持向量机 (SVM)")
    # 使用网格搜索调优 SVM 模型
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train.values.ravel())

    print("\n最佳参数：", grid_search.best_params_)
    print("最佳交叉验证准确率：", grid_search.best_score_)


# 5. 自动化数据分析
def generate_data_report(iris_data):
    profile = pp.ProfileReport(iris_data, title="Iris 数据报告")
    profile.to_file("iris_data_report.html")
    print("\n数据报告已生成：iris_data_report.html")


# 主函数
if __name__ == "__main__":
    # 加载数据
    X, y, iris_data, iris_dataset = load_and_describe_data()

    # 数据探索与可视化
    explore_and_visualize_data(X, y, iris_data)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # 模型训练与评估
    train_and_evaluate_models(X_train, X_test, y_train, y_test, iris_dataset)

    # 超参数调优
    hyperparameter_tuning(X_train, y_train)

    # 自动化数据分析
    generate_data_report(iris_data)
