import numpy as np

def comp(x):
    return x[1]

class KNN:

    # k: int,最近邻个数.
    def __init__(self, k=5):
        self.k = k

    # 此处需要填写，建议欧式距离，计算一个样本与训练集中所有样本的距离
    def distance(self, one_sample, X_train):
        distances = []
        for sample in X_train:
            distances.append(np.linalg.norm(one_sample - sample))
        return distances

    # 此处需要填写，获取k个近邻的类别标签
    def get_k_neighbor_labels(self, distances, y_train, k):
        distances = np.array(distances)
        newdistance = list(zip(y_train, distances))
        newdistance.sort(key=comp)
        labels = []
        for i in range(k):
            labels.append(newdistance[i][0])
        return labels

    # 此处需要填写，标签统计，票数最多的标签即该测试样本的预测标签
    def vote(self, one_sample, X_train, y_train, k):
        distances = self.distance(one_sample, X_train)
        labels = self.get_k_neighbor_labels(distances, y_train, self.k)
        counts = [labels.count(label) for label in labels]
        label = labels[counts.index(max(counts))]
        return label

    # 此处需要填写，对测试集进行预测
    def predict(self, X_test, X_train, y_train):
        y_pred = []
        for X in X_test:
            y_pred.append(self.vote(X, X_train, y_train, self.k))
        return y_pred


def main():
    clf = KNN(k=5)
    train_data = np.genfromtxt('./data/train_data.csv', delimiter=' ')
    train_labels = np.genfromtxt('./data/train_labels.csv', delimiter=' ')
    test_data = np.genfromtxt('./data/test_data.csv', delimiter=' ')
    # 将预测值存入y_pred(list)内
    y_pred = clf.predict(test_data, train_data, train_labels)
    np.savetxt("181220028_ypred.csv", np.array(y_pred), delimiter=' ')


if __name__ == "__main__":
    main()
    print("Finished.")
