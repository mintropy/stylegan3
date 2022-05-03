from fastapi import FastAPI


app = FastAPI()


@app.get("/api2/gen-image/")
def gen_image():
    return "Hello"
