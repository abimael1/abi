from tensorflow.keras import layers

def criar_modelo(input_shape):
    """
    Cria o modelo neural da IA Green Mind.

    Args:
        input_shape: Tupla com o tamanho da entrada da rede neural (altura, largura, canais).

    Returns:
        Modelo Keras.
    """
    modelo = Sequential()
    modelo.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    modelo.add(MaxPooling2D((2, 2)))
    modelo.add(Conv2D(64, (3, 3), activation='relu'))
    modelo.add(MaxPooling2D((2, 2)))
    modelo.add(Flatten())
    modelo.add(Dense(64, activation='relu'))
    modelo.add(Dense(1, activation='sigmoid'))
    return modelo

