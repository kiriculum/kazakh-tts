# KazakhTTS API server

### Provides REST API endpoints to synthesize human voice from a kazakh text.  

## Deployment
### Basics

* clone/copy project files
* `cd kazakh-tts`
* `pip -m venv venv`
* `source venv/bin/activate`
* `pip install -r requirements.txt`
### Start with `python3 main.py`


### Models

To add a new model(s) for the API create `kazakh-tts/models` folder and 
copy model's files inside. Each model consists of a Voice model + Vocoder model.
Example of a model structure:

```yaml
models:
  male1:
    tts_stats_raw_char:
      train:
        feats_stats.npz
    tts_train_raw_char:
      config.yaml
      train.loss.ave_5best.pth
    vocoder:
      checkpoint-400000steps.plk
      config.yml
```