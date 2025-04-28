from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import logging
from ultralytics.utils import LOGGER

LOGGER.setLevel(logging.WARNING)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routes.stream_r import stream_router
from routes.train_r import train_router

app.include_router(stream_router)
app.include_router(train_router)
