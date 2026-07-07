from fastapi import FastAPI
from fastapi.testclient import TestClient

def make_app():
    s = {"id": "1"}
    app = FastAPI()
    @app.get("/id")
    def get_id():
        return s["id"]
    @app.post("/reset")
    def reset():
        nonlocal s
        s = {"id": "2"}
        return {"ok": True}
    return app

client = TestClient(make_app())
print(client.get("/id").json())
client.post("/reset")
print(client.get("/id").json())
