from fastapi import FastAPI
import json

app = FastAPI()

@app.get("/metrics")
def metrics():
    with open("data/events.json") as f:
        events = json.load(f)

    return {
        "unique_visitors": len(events)
    }
