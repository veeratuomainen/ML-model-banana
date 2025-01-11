# ML model for banana quality recognition

This a ANN (artificial neural network) ML model used for recognizing the quality of bananas. The dataset includes different numerical variables such as size, sweetness, and ripeness, and one textual variable, quality, which is either good or bad. The dataset used is from Kaggle and can be downloaded from here:  
https://www.kaggle.com/datasets/l3llff/banana

I first preprocessed the data. I changed 'quality' variable to numerical format (0 or 1), split the data into training and testing sets and scaled part of the data. I then build a deep learning model using Tensorflow, trained it and validated on test data. The code also includes evaluating the performance of the model by calculating different metrics such as accuracy, mean absolute error (MAE) and root mean squared error (RMSE). The accuracy of the model turned out to be 96%.

I made this model in April 2024
