import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K

np.set_printoptions(precision=2)


def LoadData(DataDir, val_p=0.1, in_shape=(100,100,3)):
    Dataset = []
    Target = []

    ValData = []
    ValTarget = []

    for path, subdirs, files in os.walk(DataDir):
        for name in files:
            img = cv2.imread(path + "\\" + name, 1)
            if (img.shape == in_shape):
                Dataset.append(img)
                if name[0] == "V":
                    Target.append([1])
                else:
                    Target.append([0])

    for i in range(max(20, int(val_p * Dataset.__len__()))):
        t = randint(0, len(Dataset) - 1)
        ValData.append(Dataset[t])
        del Dataset[t]
        ValTarget.append(Target[t])
        del Target[t]

    Dataset = np.array(Dataset).astype('float32') / 255.
    Target = np.array(Target).astype('float32')

    ValData = np.array(ValData).astype('float32') / 255.
    ValTarget = np.array(ValTarget).astype('float32')
    return Dataset, Target, ValData, ValTarget


def PlotData(Dataset, pred, Target, n, m):
    for i in range(n):
        for j in range(m):
            img = Dataset[i * m + j, :, :, :]
            img = img[..., ::-1]

            plt.subplot(n, m, i * m + j + 1)
            plt.axis("off")
            plt.imshow(img)
            plt.title("Prediction: " + str(pred[i * m + j]) + " Target: " + str(Target[i * m + j]))
    plt.show()

def DefineModel(in_shape):
    x = Input(shape=(in_shape), name="input_layer")

    conv1 = Conv2D(filters=5, kernel_size=(3, 3), strides=(3, 3), padding="same")(x)
    conv1 = Activation("relu")(conv1)

    conv2 = Conv2D(filters=4, kernel_size=(3, 3), strides=(3, 3), padding="same")(conv1)
    conv2 = Activation("relu")(conv2)

    conv3 = Conv2D(filters=4, kernel_size=(3, 3), strides=(3, 3), padding="same")(conv2)
    conv3 = Activation("relu")(conv3)

    pool2 = Flatten()(conv3)

    y = Dense(1, name="linear_layer")(pool2)
    y = Activation("sigmoid")(y)

    model = Model(inputs=x, outputs=y)
    return model
def DefineModel_V2(in_shape):
    x = Input(shape=(in_shape), name="input_layer")

    conv1 = Conv2D(filters=5, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
    conv1 = Activation("relu")(conv1)

    conv2 = Conv2D(filters=4, kernel_size=(3, 3), strides=(2, 2), padding="same")(conv1)
    conv2 = Activation("relu")(conv2)

    conv3 = Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding="same")(conv2)
    conv3 = Activation("relu")(conv3)

    conv4 = Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding="same")(conv3)
    conv4 = Activation("relu")(conv4)

    pool = MaxPooling2D(5,5)(conv4)

    pool2 = Flatten()(pool)

    y = Dense(1, name="linear_layer")(pool2)
    y = Activation("sigmoid")(y)

    model = Model(inputs=x, outputs=y)
    return model
def DefineModel_V3(in_shape):
    x = Input(shape=(in_shape), name="input_layer")

    conv1 = Conv2D(filters=5, kernel_size=(3, 3), strides=(2, 2), padding="same")(x)
    conv1 = Activation("relu")(conv1)

    conv2 = Conv2D(filters=6, kernel_size=(3, 3), strides=(2, 2), padding="same")(conv1)
    conv2 = Activation("relu")(conv2)

    conv3 = Conv2D(filters=4, kernel_size=(2, 2), strides=(1, 1), padding="same")(conv2)
    conv3 = Activation("relu")(conv3)

    conv4 = Conv2D(filters=3, kernel_size=(2, 2), strides=(1, 1), padding="same")(conv3)
    conv4 = Activation("relu")(conv4)

    pool = MaxPooling2D(5, 5)(conv4)




    pool2 = Flatten()(pool)
    dense = Dense(4, name= "dense")(pool2)
    y = Dense(1, name="linear_layer")(dense)
    y = Activation("sigmoid")(y)

    model = Model(inputs=x, outputs=y)
    return model


class AI:

    def LoadData(self, Dir):
        self.Dataset, self.Target, self.ValData, self.ValTarget = LoadData(Dir)
        self.weights = {0.0: 0.15, 1.0: 0.85}

    def addModel(self, in_shape):
        self.model = DefineModel_V3(in_shape)
        self.model.summary()
        #optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True, clipnorm=1.)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["acc"], batch_normalization = True)
        self.elapsed_epochs = 0

    def saveModel(self, Dir):
        self.model.save(Dir)
        print("Model Saved")
    def loadModel(self, Dir):
        self.model = load_model(Dir)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["acc"], batch_normalization = True)
        self.model.summary()
        print("Model Loaded")
    def evaluateModel(self):
        return self.model.evaluate(self.ValData, self.ValTarget, verbose=0)
    def getKernels(self):
        ws = []
        if(self.model):
            for layer in range(self.model.layers.__len__()):
                w = self.model.get_layer(index=layer).get_weights()
                if(w.__len__()>0):
                    if(w[0].shape.__len__()>2):
                        ws.append(w[0])
            return ws
        else:
            return None
    def PlotKernels(self):
        Kernels = 0
    def trainDataGen(self, epochs, batch_size=32, spe= 64):
        self.datagen = ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0,
            height_shift_range=0,
            validation_split=0.1,
            horizontal_flip=True,
            vertical_flip=True)

        self.datagen.fit(self.Dataset)
        self.model.fit_generator(self.datagen.flow(self.Dataset, self.Target, batch_size=batch_size),
                                steps_per_epoch=spe, epochs=epochs, shuffle=True,
                                class_weight=self.weights)


        self.elapsed_epochs += epochs
    def trainStandard(self, epochs, batch_size=32):
        self.model.fit(x=self.Dataset, y=self.Target,
            batch_size=batch_size, epochs=epochs, shuffle=True,
            class_weight = self.weights,
            validation_data=(self.ValData, self.ValTarget))
        self.elapsed_epochs += epochs
    def make_prediction(self, pics):
        return self.model.predict(pics)
    def make_prediction_ValData(self):
        pred = self.model.predict(self.ValData)
        l =[]
        for p in range(pred.__len__()):
            l.append("p: " + str(pred[p])+ " t: " + str(self.ValTarget[p]))
        return l
