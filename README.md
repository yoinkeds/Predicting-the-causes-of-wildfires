# Predicting-the-causes-of-wildfires
### Link to the dataset: https://www.kaggle.com/rtatman/188-million-us-wildfires

The original code files have the predictions with all the categories. Reducing them to four different categories could give better results.

Seperating the causes of the fires into four categories:
```python 
def set_label(cat):
    cause = 0
    natural = ['Lightning']
    accidental = ['Structure','Fireworks','Powerline','Railroad','Smoking','Children','Campfire','Equipment Use','Debris Burning']
    malicious = ['Arson']
    other = ['Missing/Undefined','Miscellaneous']
    if cat in natural:
        cause = 1
    elif cat in accidental:
        cause = 2
    elif cat in malicious:
        cause = 3
    else:
        cause = 4
    return cause
 ```
