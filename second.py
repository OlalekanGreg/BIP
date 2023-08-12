import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Loading dataset
data = pd.read_csv('cleaned_data (1).csv')

# Preprocessing
# Dropping rows with missing values
data = data.dropna()

# Converting the date column to datetime and extracting additional features (e.g., month, day)
data['AE_Arrive_Date'] = pd.to_datetime(data['AE_Arrive_Date'])
data['Month'] = data['AE_Arrive_Date'].dt.month
data['Day'] = data['AE_Arrive_Date'].dt.day

# Handle age column
def handle_age(age):
    if age == '85+':
        return 85
    elif '-' in age:
        ages = age.split('-')
        return (int(ages[0]) + int(ages[1])) / 2
    else:
        return int(age)

data['Age_Band'] = data['Age_Band'].apply(handle_age)

# Convert categorical variables to numerical using Label Encoding
label_encoder = LabelEncoder()
data['Age_Band'] = label_encoder.fit_transform(data['Age_Band'])
data['AE_Arrive_Date'] = label_encoder.fit_transform(data['AE_Arrive_Date'])
data['AE_HRG'] = label_encoder.fit_transform(data['AE_HRG'])
data['ICD10_Chapter_Code'] = label_encoder.fit_transform(data['ICD10_Chapter_Code'])
data['Treatment_Function_Code'] = label_encoder.fit_transform(data['Treatment_Function_Code'])


# Define features and target variable removing Length_Of_Stay_Days, AE_Arrive_Date, 'Month', 'Day', 'Age_Band',
X = data[['IMD_Decile_From_LSOA', 'Sex','AE_Time_Mins','AE_Num_Diagnoses','AE_Num_Investigations',
          'AE_Num_Treatments','AE_Arrival_Mode', 'Age_Band',
          'Provider_Patient_Distance_Miles', 'ProvID','Record_ID',
           'AE_HRG', 'ICD10_Chapter_Code', 'Treatment_Function_Code']].values
y = data['Admitted_Flag'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification (admitted or not)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
