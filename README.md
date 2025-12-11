# Learning Models

## Dataset
The models use the public-access driving dataset POLIDriving. After feature selection, the following attributes were selected: *observation hour*, *speed*, *rpm*, *acceleration*, *throttle position*, *engine temperature*, *engine load value*, *heart rate*, *current weather*, *visibility*, *precipitation*, *accidents onsite*, *design speed*, *accidents time*, and *risk level*. 

## Libraries
Those models use <a href=https://scikit-learn.org/>scikit-learn</a> and <a href=https://www.tensorflow.org/>tensor flow</a>.

## Models
The following machine learning models were created.

* Gradient boosting machines (GBM)
* Support vector machines (SVM)
* Convolutional neural networks (CNN)
* Radial basis function networks (RBFNN) and
* Multilayer perceptrons (MLP)

## Configurations



## Data file format

Data files contain the following attributes.

|#|Attribute|Class|Units|Data source|
|-|---|---|---|---|
|1|time|Timestamp||Vehicle data|
|2|speed|Numeric|km/h|Vehicle data|
|3|revolutions per minute|Numeric|rpm|Vehicle data|
|4|acceleration|Numeric|m/s2|Vehicle data|
|5|throttle position|Numeric|%|Vehicle data|
|6|engine temperature|Numeric|C|Vehicle data|
|7|system voltage|Numeric|volts|Vehicle data|
|8|distance traveled|Numeric|km|Vehicle data|
|9|engine load value|Numeric|%|Vehicle data|
|10|latitude|Numeric||Vehicle data|
|11|longitude|Numeric||Vehicle data|
|12|altitude|Numeric|m|Vehicle data|
|13|id vehicle|Numeric||Vehicle data|
|14|heart rate|Numeric|bpm|Driver's data|
|15|body temperature|Numeric|C|Driver's data|
|16|id driver|Numeric||Driver's data|
|17|current weather|Categorical||Weather data|
|18|has precipitation|Boolean||Weather data|
|19|is day time|Boolean||Weather data|
|20|temperature|Numeric|C|Weather data|
|21|wind speed|Numeric|km/h|Weather data|
|22|wind direction|Numeric||Weather data|
|23|relative humidity|Numeric|%|Weather data|
|24|visibility|Numeric|km|Weather data|
|25|uv index|Numeric||Weather data|
|26|cloud cover|Numeric||Weather data|
|27|ceiling|Numeric|m|Weather data|
|28|pressure|Numeric|mb|Weather data|
|29|precipitation|Numeric|mm|Weather data|
|30|accidents on site|Numeric|deaths|Traffic accidents|
|31|design speed|Numeric|km/h|Road geometrics characteristics|
|32|accidents time|Numeric|deaths|Road geometrics characteristics|

# Publication

If you use POLIDriving in your research, please cite it as follows.

@article{marcillo2024polidriving,<br>
  title={POLIDriving: A Public-Access Driving Dataset for Road Traffic Safety Analysis},<br>
  author={Marcillo, Pablo and Arciniegas-Ayala, Cristian and Valdivieso Caraguay, {\'A}ngel Leonardo and Sanchez-Gordon, Sandra and Hern{\'a}ndez-{\'A}lvarez, Myriam},<br>
  journal={Applied Sciences},
  volume={14},<br>
  number={14},<br>
  pages={6300},<br>
  year={2024},<br>
  publisher={MDPI}<br>
}

# Downloads

The size of POLIDriving is about 150 MB.

# Contact

For questions or suggestions, please contact pablomarcillolar@gmail.com or pablo.marcillo@epn.edu.ec
