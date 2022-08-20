# coding: utf-8
# Copyright (C) 2022, [Breezedeus](https://github.com/breezedeus).
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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
    image = image.file
    image = Image.open(image).convert('RGB')
    res = OCR_MODEL.ocr(image)
    for _one in res:
        _one['position'] = _one['position'].tolist()
        if 'cropped_img' in _one:
            _one.pop('cropped_img')

    return OcrResponse(results=res).dict()
