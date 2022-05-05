from fastapi import FastAPI
from fastapi.responses import FileResponse


app = FastAPI()


@app.get("/api/gen-image/")
def gen_image():
    file_dir = "test.png"
    return FileResponse(file_dir)
