from loader import DataLoader
from perceptron import Neuron

lr: float = 0.1


def main() -> None:
    train_data = DataLoader(5).load_data()
    perceptron = Neuron(len(train_data[0].data))
    perceptron.train(train_data=train_data, learning_rate=lr)

    for _ in range(10):
        print(f'Input {train_data[_].number}, got:  {perceptron.predict(train_data[_].data)}')

if __name__ == '__main__':
    main()
