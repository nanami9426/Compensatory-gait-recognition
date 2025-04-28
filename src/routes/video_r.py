import torch
from torch import nn
import os
import pandas as pd
from fastapi import Response, APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from nets.net import Pnet
from conf import window_size, hidden_size, num_layers
from pathlib import Path
from datetime import datetime

video_router = APIRouter()