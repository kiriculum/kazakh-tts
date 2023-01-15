# KazakhTTS API server

### Provides REST API endpoints to synthesize human voice from a kazakh text.  

## Deployment
### Basics

* clone/copy project files
* `cd kazakh-tts & setup.sh`



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


## Start
* use the starting script e.g. `./start.sh`