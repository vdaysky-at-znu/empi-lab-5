from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def main():
    model = LinearRegression()
    xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    ys = [-42, - 39, - 36, - 33, - 30, - 27, - 24, - 21, - 18, - 15, - 12, - 9, - 6, - 3, 0, 3, 6, 9, 12, 15]
    model.fit([[x] for x in xs], [[y] for y in ys])

    test_data = [[20], [21], [22]]
    predicted = model.predict(test_data)

    plt.plot(xs, ys, color='red')
    plt.plot(test_data, predicted, color='green')
    plt.show()


if __name__ == '__main__':
    main()
