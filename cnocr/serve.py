# coding: utf-8
from copy import deepcopy
from typing import List, Dict, Any

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile
from PIL import Image

from cnocr import CnOcr
from cnocr.utils import set_logger

logger = set_logger(log_level='DEBUG')

app = FastAPI()
OCR_MODEL = CnOcr()


class OcrResponse(BaseModel):
    status_code: int = 200
    results: List[Dict[str, Any]]

    def dict(self, **kwargs):
        the_dict = deepcopy(super().dict())
        return the_dict


@app.get("/")
async def root():
    return {"message": "Welcome to CnOCR Server!"}


@app.post("/ocr")
async def ocr(image: UploadFile) -> Dict[str, Any]:
    image = Image.open(image.file).convert('RGB')
    res = OCR_MODEL.ocr(image)
    for _one in res:
        _one['position'] = _one['position'].tolist()
        if 'cropped_img' in _one:
            _one.pop('cropped_img')

    return OcrResponse(results=res).dict()
