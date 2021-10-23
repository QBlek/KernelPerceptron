import pandas as pd
import numpy as np
import KernelizedPerceptron as kpa
# import kptest as kpa
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Dataset setting
    dataset = pd.read_csv('./data/fashion-mnist_train.csv')
    testset = pd.read_csv('./data/fashion-mnist_test.csv')

    labels_train = dataset['label']
    images_train = dataset.iloc[:, 1:]

    x_temp = images_train.values
    y_temp = labels_train.values

    index_list = np.arange(0, dataset.shape[0])
    train_length = int(0.8 * len(index_list))
    train_indices = index_list[:train_length]
    validation_indices = index_list[train_length:]

    """
    x_train = x_temp[train_indices]
    y_train = y_temp[train_indices]
    x_validation = x_temp[validation_indices]
    y_validation = y_temp[validation_indices]
    """
    x_train = x_temp[:4000]
    y_train = y_temp[:4000]
    x_validation = x_temp[4000:5000]
    y_validation = y_temp[4000:5000]


    labels_test = testset['label']
    images_test = testset.iloc[:, 1:]
    x_testT = images_test.values
    y_testT = labels_test.values

    x_test = x_testT[:1000]
    y_test = y_testT[:1000]

    # Learning
    kpm = kpa.KernelizedPerceptron(n_iter=5)

    # run the training set
    print("Training Set")
    result_train = kpm.fit(x_train, y_train, 0, 2)
    w_train = sum(result_train[3], [])

    # run the validation set
    print("Validation Set")
    result_validation = kpm.fit(x_validation, y_validation, w_train, 2)
    w_validation = sum(result_validation[3], [])

    # run the test set
    print("Test Set")
    result_test = kpm.fit(x_test, y_test, w_validation, 2)

    # Visualization
    plt.plot(result_train[0], result_train[2], label='Training_accuracy')
    plt.plot(result_validation[0], result_validation[2], label='Validation_accuracy')
    plt.plot(result_test[0], result_test[2], label='Test_accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

