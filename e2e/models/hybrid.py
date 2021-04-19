from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras.applications import InceptionResNetV2                                                 #MODIFYput                                 #MODIFY
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD                                           #MODIFY
from tensorflow.keras.losses import categorical_crossentropy

def model(weight_path):
    model = InceptionResNetV2(include_top = False,
                weights="imagenet",
                input_tensor=None,
                input_shape=(256,256,3),
                pooling=None,
                classes=40)

    final_model = Sequential()
    final_model.add(model)
    final_model.add(Flatten())
    final_model.add(Dropout(0.2))
    final_model.add(Dense(units=2048,activation="relu"))
    final_model.add(Dropout(0.2))
    final_model.add(Dense(units=2048,activation="relu"))
    final_model.add(Dense(units=40, activation="softmax"))

    opt = SGD(lr=1e-4, momentum=0.9)
    final_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    final_model.load_weights(weight_path)
    # final_model = load_model(weight_path)

    return final_model