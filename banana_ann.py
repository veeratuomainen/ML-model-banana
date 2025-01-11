import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# luetaan data
df = pd.read_csv('banana_quality.csv')

# muutetaan Quality muuttuja numeeriseksi dataksi
label_encoder = LabelEncoder()
df['QualityNumeral'] = label_encoder.fit_transform(df['Quality'])

# X ja y datasetit
X = df.iloc[:, 0:7]
y = df.loc[:, ['QualityNumeral']]

# jaetaan train ja test datasetteihin
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=25, random_state=0)

# skaalataan X
x_scaler = MinMaxScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

# opetetaan malli
model = Sequential()
model.add(Dense(50, input_dim=X.shape[1], activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test,y_test))

# visualisoidaan mallin oppiminen
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

# tehdään ennusteet
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
ax = plt.axes()
sns.heatmap(cm, ax = ax, annot=True, fmt='g')
plt.show()

# metriikat
acc = accuracy_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'Accuracy score: {acc}')
print(f'r2:  {round(r2,4)}')
print(f'mae: {round(mae,4)}')
print(f'rmse: {round(rmse,4)}')

