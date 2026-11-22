from src.dataset import training_data
from src.perceptron import Perceptron

model = Perceptron()
model.train(training_data)

prediction = model.predict([9, 1])

print(f"Prediccion para [9, 1]: {prediction}")
print(f"Pesos aprendidos: {model.weights}")
print(f"Bias aprendido: {model.bias}")

