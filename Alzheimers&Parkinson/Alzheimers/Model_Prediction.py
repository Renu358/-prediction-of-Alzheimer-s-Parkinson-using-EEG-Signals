import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report,confusion_matrix

model = load_model('Project_model.h5') 
new_data = pd.read_csv('project_test.csv')

X_new = new_data.drop(columns=['target'])  
y_test=new_data['target']
y_pred_new = np.argmax(model.predict(X_new), axis=-1)
print(y_pred_new)
conf_matrix = confusion_matrix(y_test, y_pred_new)
print("Confusion Matrix:")
correctly_classified_counts = np.diag(conf_matrix)
class_names = ['Healthy', 'Alzheimers', 'Parkinsons']

for class_name, count in zip(class_names, correctly_classified_counts):
    print(f"Correctly classified {class_name}: {count}")
class_report = classification_report(y_test, y_pred_new)
print("\nClassification Report:")
print(class_report)
