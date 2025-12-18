# Get Started 

## Enviroment

**Python=3.8**

**Packages**:
- torch==1.8.1+culll
- torchaudio==0.8.1
- torchvision==0.9.1+culll
- numpy==1.19.2
- opencv-python==3.4.0.12
- pandas==1.1.5
- Pillow==8.4.0
- scipy==1.4.1
- pytz==2024.2
- PyYAML==6.0.1

(The above environment configuration is not the optimal result, other environments versions might work too.)


## datasets
The link to the reconstructed DAGM2007* dataset is as follows: [DAGM2007*](https://www.kaggle.com/datasets/amor000/reconstructed-dagm2007-dataset).

## Run

Please specity dataset path and log foder before running. Please execute the following command to start the training:
```
python tools/train.py config/tinyvit.py
```

## Acknowledgement

Thank for great inspiration from [PatchCore](https://github.com/Fafa-DL/Awesome-Backbones)
