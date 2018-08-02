#

from keras import layers, models
from DeskImgDataSet import DeskImgDataSet


def buildModel():

    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512,activation="relu"))
    model.add(layers.Dense(10,activation="sigmoid")) #分類先の種類分設定

    #モデル構成の確認
    model.summary()


    return model


def training(model):
    pass


def main():
    model = buildModel()


if __name__ == "__main__":
    main()