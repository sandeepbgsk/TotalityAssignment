import warnings

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

if __name__ == '__main__':
    train_path = r'C:\Users\bhgsk\PycharmProjects\Totality_Assignment\dog-breed-identification\train'
    valid_path = r'C:\Users\bhgsk\PycharmProjects\Totality_Assignment\dog-breed-identification\test'

    resnet = ResNet50(include_top=False, weights="imagenet")

    for layer in resnet.layers:
        layer.trainable = False

    # Flatten last but one layer
    x = GlobalAveragePooling2D()(resnet.output)

    # Connect last but one layer to 120 prediction nodes
    prediction = Dense(120, activation='softmax')(x)

    # create a model object
    custom_resnet = Model(inputs=resnet.input, outputs=prediction)

    # Complie the model
    custom_resnet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Now that the bare model is ready, do data preparation
    # https://vijayabhaskar96.medium.com/tutorial-on-keras-flow-from-dataframe-1fd4493d237c

    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import pandas as pd


    def append_ext(fn):
        return fn + ".jpg"


    traindf = pd.read_csv(r"C:\Users\bhgsk\PycharmProjects\Totality_Assignment\dog-breed-identification\labels.csv",
                          dtype=str)
    testdf = pd.read_csv(
        r"C:\Users\bhgsk\PycharmProjects\Totality_Assignment\dog-breed-identification\sample_submission.csv", dtype=str)
    traindf["id"] = traindf["id"].apply(append_ext)
    testdf["id"] = testdf["id"].apply(append_ext)

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       validation_split=0.25)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_dataframe(dataframe=traindf,
                                                     directory=r"C:\Users\bhgsk\PycharmProjects\Totality_Assignment\dog-breed-identification\train",
                                                     x_col="id",
                                                     y_col="breed",
                                                     shuffle=True,
                                                     target_size=(224, 224),
                                                     batch_size=32,
                                                     class_mode='categorical')

    valid_set = train_datagen.flow_from_dataframe(dataframe=traindf,
                                                  directory=r"C:\Users\bhgsk\PycharmProjects\Totality_Assignment\dog-breed-identification\train",
                                                  x_col="id",
                                                  y_col="breed",
                                                  shuffle=True,
                                                  target_size=(224, 224),
                                                  batch_size=32,
                                                  class_mode='categorical')

    test_set = test_datagen.flow_from_dataframe(dataframe=testdf,
                                                directory=r"C:\Users\bhgsk\PycharmProjects\Totality_Assignment\dog-breed-identification\test",
                                                x_col="id",
                                                y_col=None,
                                                batch_size=32,
                                                seed=42,
                                                shuffle=False,
                                                class_mode=None,
                                                target_size=(224, 224))
    # Fit the custom resnet model with above dataset
    r = custom_resnet.fit_generator(
        training_set,
        validation_data=valid_set,
        epochs=5,
        steps_per_epoch=len(training_set),
        validation_steps=len(valid_set)
    )

    # loss
    plt.plot(r.history['loss'], label='train loss')
    plt.plot(r.history['val_loss'], label='val loss')
    plt.legend()
    plt.show()
    plt.savefig('LossVal_loss')

    # accuracies
    plt.plot(r.history['acc'], label='train acc')
    plt.plot(r.history['val_acc'], label='val acc')
    plt.legend()
    plt.show()
    plt.savefig('AccVal_acc')

    custom_resnet.save('dog_classification.h5')
