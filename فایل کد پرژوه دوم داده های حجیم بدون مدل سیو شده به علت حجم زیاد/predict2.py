from fastapi import APIRouter, FastAPI, File, UploadFile, Response
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import io
from PIL import Image
import logging
import os
import sys

# تنظیمات خروجی برای پشتیبانی از کدگذاری UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

router = APIRouter()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# تنظیمات مدل و داده‌ها
NUM_CLASSES = 4
INPUT_HEIGHT = 128
INPUT_WIDTH = 128
LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 16
SHUFFLE = True
MODEL_WEIGHTS_FILE = 'fcn32s_model.weights.h5'  # تغییر نام فایل وزن‌ها

AUTOTUNE = tf.data.AUTOTUNE

(train_ds, valid_ds, test_ds) = tfds.load(
    "oxford_iiit_pet",
    split=["train[:85%]", "train[85%:]", "test"],
    batch_size=BATCH_SIZE,
    shuffle_files=SHUFFLE,
)

def unpack_resize_data(section):
    image = section["image"]
    segmentation_mask = section["segmentation_mask"]
    resize_layer = keras.layers.Resizing(INPUT_HEIGHT, INPUT_WIDTH)
    image = resize_layer(image)
    segmentation_mask = resize_layer(segmentation_mask)
    return image, segmentation_mask

train_ds = train_ds.map(unpack_resize_data, num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.map(unpack_resize_data, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(unpack_resize_data, num_parallel_calls=AUTOTUNE)

input_layer = keras.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3))
vgg_model = keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 3))

fcn_backbone = keras.models.Model(
    inputs=vgg_model.input,
    outputs=[
        vgg_model.get_layer(block_name).output
        for block_name in ["block3_pool", "block4_pool", "block5_pool"]
    ],
)

fcn_backbone.trainable = False

x = fcn_backbone(input_layer)
units = [4096, 4096]
dense_convs = []

for filter_idx in range(len(units)):
    dense_conv = keras.layers.Conv2D(
        filters=units[filter_idx],
        kernel_size=(7, 7) if filter_idx == 0 else (1, 1),
        strides=(1, 1),
        activation="relu",
        padding="same",
        use_bias=False,
        kernel_initializer='he_normal',
    )
    dense_convs.append(dense_conv)
    dropout_layer = keras.layers.Dropout(0.5)
    dense_convs.append(dropout_layer)

dense_convs = keras.Sequential(dense_convs)
dense_convs.trainable = False

x = dense_convs(x[-1])

pool5 = keras.layers.Conv2D(
    filters=NUM_CLASSES,
    kernel_size=(1, 1),
    padding="same",
    strides=(1, 1),
    activation="relu",
)(x)

fcn32s_conv_layer = keras.layers.Conv2D(
    filters=NUM_CLASSES,
    kernel_size=(1, 1),
    activation="softmax",
    padding="same",
    strides=(1, 1),
)

fcn32s_upsampling = keras.layers.UpSampling2D(
    size=(32, 32),
    data_format=keras.backend.image_data_format(),
    interpolation="bilinear",
)

final_fcn32s_pool = pool5
final_fcn32s_output = fcn32s_conv_layer(final_fcn32s_pool)
final_fcn32s_output = fcn32s_upsampling(final_fcn32s_output)

fcn32s_model = keras.Model(inputs=input_layer, outputs=final_fcn32s_output)

fcn32s_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

if os.path.exists(MODEL_WEIGHTS_FILE):
    fcn32s_model.load_weights(MODEL_WEIGHTS_FILE)
    logger.info(f"Model weights loaded from {MODEL_WEIGHTS_FILE}")
else:
    fcn32s_model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
    )
    fcn32s_model.save_weights(MODEL_WEIGHTS_FILE)
    logger.info(f"Model weights saved to {MODEL_WEIGHTS_FILE}")

test_loss, test_accuracy = fcn32s_model.evaluate(test_ds)
logger.info(f"Test Loss: {test_loss}")
logger.info(f"Test Accuracy: {test_accuracy}")

@router.post("/predict1/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image = image.resize((INPUT_WIDTH, INPUT_HEIGHT))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = fcn32s_model.predict(image)[0]
    segmentation_mask = np.argmax(prediction, axis=-1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(image[0])
    plt.subplot(1, 2, 2)
    plt.title("Segmentation Mask")
    plt.imshow(segmentation_mask, cmap="inferno")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")

app = FastAPI()
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


