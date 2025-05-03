import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import tensorflow as tf
import warnings

from starlette.responses import HTMLResponse, FileResponse

warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI()

from keras.layers import TFSMLayer

MODEL = TFSMLayer("models/1", call_endpoint="serving_default")

CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']


def read_files_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_files_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


@app.get("/")
async def index():
    return FileResponse("index.html")


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
