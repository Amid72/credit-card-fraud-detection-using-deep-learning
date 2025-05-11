#!/usr/bin/env python
# coding: utf-8

# Data Loading & Preprocessing

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Load the dataset
df = pd.read_csv("creditcard.csv")


# In[3]:


print(df.head())


# In[4]:


# Display the shape and class distribution
print(df.shape)
print(df["Class"].value_counts())


# In[5]:


# Visualize class distribution
sns.countplot(data=df, x="Class")
plt.title("Fraud vs Non-Fraud Distribution")
plt.show()


# SMOTE Balancing

# In[6]:


from imblearn.over_sampling import SMOTE
from collections import Counter


# In[7]:


# Define X and y
X = df.drop('Class', axis=1)
y = df['Class']


# In[8]:


# Apply SMOTE to balance classes
X_res, y_res = SMOTE().fit_resample(X, y)
print(Counter(y_res))


# In[9]:


# Visualize the balanced class distribution
sns.countplot(x=y_res)
plt.title("After SMOTE: Class Distribution")
plt.show()


# Data Reshaping

# In[10]:


def reshape_data(X_train, X_test, batch_size=30):
    # Check the size of the data
    print(f"Training data size: {X_train.values.size}")
    
    # Ensure that the size of the data is divisible by batch_size
    if X_train.values.size % batch_size == 0:
        X_train_cnn = X_train.values.reshape(-1, batch_size, 1)
        X_test_cnn = X_test.values.reshape(-1, batch_size, 1)
    else:
        # Adjust reshaping by trimming to ensure the size is divisible by batch_size
        trim_size = (X_train.values.size // batch_size) * batch_size
        X_train_cnn = X_train.values[:trim_size].reshape(-1, batch_size, 1)
        X_test_cnn = X_test.values[:trim_size].reshape(-1, batch_size, 1)
    
    return X_train_cnn, X_test_cnn

# Split the data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Reshape the data for CNN input
X_train_cnn, X_test_cnn = reshape_data(X_train, X_test)


# CNN Model Definition

# In[11]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import Adam


# In[12]:


def create_cnn_model(input_shape):
    model = Sequential()
    
    # Add a 1D convolutional layer
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    
    # Add a max pooling layer
    model.add(MaxPooling1D(pool_size=2))
    
    # Flatten the output to connect it to the dense layer
    model.add(Flatten())
    
    # Add a dense layer with a softmax activation function for classification
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Since it's binary classification
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# Model Training (CNN)

# In[13]:


# Define input shape based on the reshaped data (for example: 30 time steps and 1 feature)
input_shape = (30, 1)  # Adjust based on your data shape

# Create and compile the CNN model
cnn_model = create_cnn_model(input_shape)

# Train the CNN model
cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_data=(X_test_cnn, y_test))


# Model Evaluation

# In[14]:


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


# In[15]:


# Get predictions from the model
y_pred_cnn = cnn_model.predict(X_test_cnn) > 0.5  # For binary classification (threshold 0.5)

# Print classification report
print(classification_report(y_test, y_pred_cnn))

# Confusion Matrix
print(confusion_matrix(y_test, y_pred_cnn))

# ROC AUC Score
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_cnn)}")


# In[16]:


from sklearn.metrics import ConfusionMatrixDisplay

# Display confusion matrix for CNN
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_cnn)
plt.title("CNN Confusion Matrix")
plt.show()


# LSTM Model Training

# In[19]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense


# In[20]:


# 🛡️ Ensure input is shaped (samples, timesteps, features) = (n, 1, 30)
X_train_lstm = X_train.values.reshape(-1, 1, 30)
X_test_lstm = X_test.values.reshape(-1, 1, 30)

# 🧠 Build LSTM Model
lstm_model = Sequential([
    LSTM(64, input_shape=(1, 30), return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# ⚙️ Compile Model
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🚀 Train Model
lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=64, validation_split=0.2)


# In[21]:


cnn_eval = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)
cnn_accuracy = cnn_eval[1]

lstm_eval = lstm_model.evaluate(X_test_lstm, y_test, verbose=0)
lstm_accuracy = lstm_eval[1]


# In[22]:


# 📊 Plot
import matplotlib.pyplot as plt
models = ['CNN', 'LSTM']
accuracies = [cnn_accuracy, lstm_accuracy]

plt.bar(models, accuracies, color=['skyblue', 'salmon'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0.9, 1.0)
plt.show()


# Evaluation Metrices

# In[27]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report


# In[28]:


def evaluate_scores(model, X_test, y_test, model_name):
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"\n📊 {model_name} Evaluation Scores")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"ROC AUC Score: {auc:.4f}")

    return acc, prec, rec, f1, auc

# 🔍 Evaluate both models
cnn_scores = evaluate_scores(cnn_model, X_test_cnn, y_test, "CNN")
lstm_scores = evaluate_scores(lstm_model, X_test_lstm, y_test, "LSTM")


# In[29]:


import matplotlib.pyplot as plt
import numpy as np


# In[30]:


# Use scores from previous evaluation
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
cnn_vals = list(cnn_scores)
lstm_vals = list(lstm_scores)

# Normalize for radar chart (0-1 scale assumed)
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
cnn_vals += cnn_vals[:1]
lstm_vals += lstm_vals[:1]
angles += angles[:1]

# Create radar chart
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
ax.plot(angles, cnn_vals, 'o-', linewidth=2, label='CNN')
ax.fill(angles, cnn_vals, alpha=0.25)

ax.plot(angles, lstm_vals, 'o-', linewidth=2, label='LSTM')
ax.fill(angles, lstm_vals, alpha=0.25)

ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_title("Model Performance Comparison", size=15)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.show()


# In[ ]:




