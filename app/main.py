from fastapi import FastAPI
from app.api.v1.routes.upload import router as upload_router
from app.utils.constants import BASE_PREFIX

app = FastAPI()


@app.get("/")
def greetUser():
    return {"message": "Hello From Smart Dustbin."}


app.include_router(upload_router, prefix=f"/upload", tags=["Uploads"])
