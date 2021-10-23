import numpy as np

class KernelizedPerceptron:

    def __init__(self, eta=1, n_iter=5):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x, y, weight, poly):
        # print("Initial weight: ", weight)
        weight_label = []
        # if it is training then initial weight == 0, but test case we have weight vector already
        if weight == 0:
            for _ in range(10):
                weight_label.append(np.zeros(len(x)))
        else:
            weight_label = weight
        iter_num = 0

        # result will include num of mistake, accuracy, final weight
        result = []
        for _ in range(4):
            result.append([])

        # iteration
        for _ in range(self.n_iter):
            iter_num += 1
            mis = 0
            no_mis = 0
            print("Iteration ", iter_num)

            num_ = 0
            for xt, yt in zip(x, y):
                score = np.zeros(10)
                cls = 0

                # score per class
                for i in range(10):
                    for j in range(len(x)):
                        score[i] += (np.inner(weight_label[i][j], (np.inner(xt, x[j]) + 1)**poly))
                        # print(i, " ", j, " ", score)
                    cls += 1

                prediction = np.array(score).argmax()

                # print("yt= ", yt, ", prediction= ", prediction)

                if yt != prediction:
                    # correct label
                    weight_label[yt][num_] = 1
                    # wrong label
                    weight_label[prediction][num_] = -1
                    mis += 1
                else:
                    no_mis += 1

                num_ += 1
                print("t= ", num_)

            print("Finished Iteration")
            accuracy = no_mis / (mis + no_mis)

            result[0].append(iter_num)
            result[1].append(mis)
            result[2].append(accuracy)

        result[3].append(weight_label)

        return result
