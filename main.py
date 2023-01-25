import re

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.security import APIKeyQuery

import config
from api_logging import logger
from number_spell import transform
from synthesize import process_text, available_models, ModelDontExist, check_voice_cache

app = FastAPI()


def auth(token: str = Depends(APIKeyQuery(name='token'))) -> str:
    if token not in config.api_tokens:
        e = HTTPException(401)
        logger.info(e)
        raise e
    return token


@app.get('/synthesize/')
async def synthesize(text: str, model: str, _token: str = Depends(auth)) -> FileResponse:
    text = re.sub(r'\d+', lambda x: transform(int(x.group())), text)
    try:
        all_models = available_models()
    except ModelDontExist as e:
        logger.info(str(e))
        raise HTTPException(400, e)
    if model not in all_models:
        e = HTTPException(404, f'Model \'{model}\' not found')
        logger.info(str(e))
        raise e
    file_path = check_voice_cache(text, model)
    if file_path:
        logger.info(f'Voice audio found in cache with model \'{model}\' for text: \'{text}\'')
    else:
        file_path, rtf = process_text(text, model)
        logger.info(f'Voice audio synthesized with model \'{model}\' for text: \'{text}\', RTF={rtf}')
    return FileResponse(file_path)


@app.get('/voices/')
async def voices(_token: str = Depends(auth)) -> list[str]:
    try:
        return sorted(available_models())
    except ModelDontExist as e:
        raise HTTPException(400, e)


if __name__ == '__main__':
    # Start uvicorn server with tts app served
    uvicorn.run(
        'main:app',
        host=config.host,
        port=config.port,
        reload=True
    )
