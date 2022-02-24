from neuron import Neuron
from loader import *

GOAL = 9
data = parse_json_data(DATA_FILE_PATH)
print(data[str(GOAL)])
neuron_1 = Neuron(35, -35)
for i in range(100):
    neuron_1.train(data, goal=GOAL)
print("Training number:", GOAL)
print("Test (input|output):", *neuron_1.test(data))
print("Neuron weights:", *neuron_1.weights)




