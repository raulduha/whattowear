import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from colorthief import ColorThief
import matplotlib.pyplot as plt
# Estamos viendo el color pero hay que enfocarlo en la prenda 
# Cargar datos de Fashion MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalizar imágenes y convertir etiquetas a one-hot encoding
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)  # 10 clases en Fashion MNIST
y_test = to_categorical(y_test, 10)

# Redimensionar imágenes para que coincidan con el formato de entrada de la red neuronal
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Definir el modelo de red neuronal convolucional (CNN)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Evaluar el modelo en el conjunto de prueba
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Guardar el modelo entrenado
model.save('whattocolour.h5')

# Cargar el modelo entrenado
model = load_model('whattocolour.h5')

# Cargar y preprocesar tus imágenes
# Asegúrate de que 'tu_imagen' sea reemplazado con la ruta a tu propia imagen
imagen_path = 'shile.jpg'
imagen_original = imageio.imread(imagen_path)

# Preprocesar la imagen
imagen = cv2.cvtColor(imagen_original, cv2.COLOR_RGB2GRAY)  # Convertir a escala de grises si es necesario
imagen = cv2.resize(imagen, (28, 28))
imagen = imagen / 255.0  # Normalizar

# Reshape la imagen para que coincida con el formato de entrada del modelo
imagen = imagen.reshape(1, 28, 28, 1)

# Hacer predicciones
prediccion = model.predict(imagen)
etiqueta_predicha = np.argmax(prediccion)

# Mapear la etiqueta predicha a la clase correspondiente (si tienes un diccionario de clases)
clases = {0: 'Camiseta', 1: 'Pantalón', 2: 'Jersey', 3: 'Vestido', 4: 'Abrigo', 5: 'Sandalia', 6: 'Camisa', 7: 'Zapatilla', 8: 'Botines'}
nombre_clase_predicha = clases[etiqueta_predicha]

# Identificar el color utilizando ColorThief
color_thief = ColorThief(imagen_path)
dominant_color = color_thief.get_color(quality=1)

# Visualizar la imagen original con la etiqueta y el color dominante
plt.imshow(imagen_original)
plt.title(f'Predicción: {nombre_clase_predicha}\nColor dominante: {dominant_color}')
plt.axis('off')
plt.show()