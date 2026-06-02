import numpy as np


def main():
    matrix = np.zeros((9, 3))
    matrix[3,1] = 1
    print(np.argmax(matrix[3]))


if __name__ == "__main__":
    main()
