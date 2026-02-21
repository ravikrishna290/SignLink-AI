from fastapi import FastAPI, UploadFile, File
import tempfile

from app.utils import extract_from_video
from app.predictor import predict_all

app = FastAPI()

@app.get("/")
def home():
    return {"message": "ISL Multi-Model Backend Running ✅"}


@app.post("/predict")
async def predict(video: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await video.read())
        video_path = tmp.name

    features = extract_from_video(video_path)

    if features is None:
        return {"error": "No landmarks detected"}

    results = predict_all(features)

    return {
        "status": "success",
        "predictions": results
    }
