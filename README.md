# facial expression recognition

Using tensorflow

## Dataset

### FER-2013
![FER-2013](data/dataset-cover.png "FER-2013")

Available on [**kaggle**](https://www.kaggle.com/datasets/msambare/fer2013)  
**Volume:** 36k images (29k train + 3.5k test)  
**Image parameters:** grayscale, 48x48  
**Categories:** 6 *(originally 7)*  

## Implementation

1. [Environment setup](impl/README.md)
2. Run training
```bash
cd impl
./run.sh
```
3. Run single predict on a pretrained model
```bash
cd impl
python3 single_predict.py --image image.png --model model.keras
```
