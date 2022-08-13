import keras
import numpy as np

x = np.random.rand(10, 2)
y = np.random.rand(10, 2)

model = keras.Sequential([
    keras.layers.Input(shape=(2)),
    keras.layers.Dense(2),
    keras.layers.Dense(2)
])

model.compile(loss='mse', optimizer='adam')

print("-- RUN 1")
model.layers[0].trainable = False
print(model.summary())  # None trainable params: 6
print(model.layers[0].weights)
model.fit(x, y, epochs=5)
print(model.layers[0].weights)    # However weights change

print("-- RUN 2")
model.layers[0].trainable = False
model.compile(loss='mse', optimizer='adam')  # need to recompile
print(model.summary())  # None trainable params: 6
print(model.layers[0].weights)
model.fit(x, y, epochs=5)
print(model.layers[0].weights)

# !!! AFTER CHANGING .trainable ATTRIBUTE NEED TO RECOMPILE MODEL !!!
