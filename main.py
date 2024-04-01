import numpy as np
import os
import librosa  # For audio processing
import noisereduce as nr
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM
from tensorflow.keras import optimizers
from tensorflow.keras.losses import BinaryCrossentropy

def preprocess_audio(audio_file):
    audio, sample_rate = librosa.load(audio_file)
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate).mean()
    rms = librosa.feature.rms(y=audio).mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate).mean()
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate).mean()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio).mean()
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20).mean(axis=1)
    
    return [chroma_stft, rms, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate, *mfccs]

import pandas as pd
import librosa

def load_dataset(audio_files_AI, audio_files_human):
    # Initialize lists to store features and labels
    features = []
    labels = []
    
    # Load and preprocess AI-generated audio samples
    for audio_file in audio_files_AI:
        audio, sample_rate = librosa.load(audio_file)
        chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate).mean()
        rms = librosa.feature.rms(y=audio).mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate).mean()
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate).mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio).mean()
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20).mean(axis=1)
        
        # Append features to the list
        features.append({
            'chroma_stft': chroma_stft,
            'rms': rms,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'rolloff': rolloff,
            'zero_crossing_rate': zero_crossing_rate,
            **{f'mfcc{i+1}': mfccs[i] for i in range(20)}
        })
        
        # Label 0 for AI-generated audio
        labels.append(0)
    
    # Load and preprocess human voice audio samples
    for audio_file in audio_files_human:
        audio, sample_rate = librosa.load(audio_file)
        chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sample_rate).mean()
        rms = librosa.feature.rms(y=audio).mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate).mean()
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate).mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio).mean()
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20).mean(axis=1)
        
        # Append features to the list
        features.append({
            'chroma_stft': chroma_stft,
            'rms': rms,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'rolloff': rolloff,
            'zero_crossing_rate': zero_crossing_rate,
            **{f'mfcc{i+1}': mfccs[i] for i in range(20)}
        })
        
        # Label 1 for human voice
        labels.append(1)
    
    # Convert lists to pandas DataFrame
    df = pd.DataFrame(features)
    df['label'] = labels
        
    # Save DataFrame to CSV file
    df.to_csv('audio_dataset.csv', index=False)
def read_folders(folder_paths):
    all_files = []
    for folder_path in folder_paths:
        files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('wav') or file.endswith('mp3')]  # Adjust file extension as needed
        all_files.extend(files)
    return all_files
# Create dataset
if __name__ == "__main__":
    # Step 1: Preprocessing and Feature Extraction
    # (This step may involve additional processing based on your feature extraction techniques)
    notebook_dir = os.getcwd()
    fake_folder_path = os.path.join(notebook_dir, "AUDIO", "REAL")
    real_folder_path = os.path.join(notebook_dir, "AUDIO", "FAKE")
    audio_files_AI = read_folders([fake_folder_path])
    audio_files_human = read_folders([real_folder_path])

    load_dataset(audio_files_AI,audio_files_human)
df=pd.read_csv("audio_dataset.csv")
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])
X = df.drop(columns=['label'])
y = df['label']
X = X.values
y = y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True ,random_state=42)
X[1]
X_reshaped = np.zeros((X_train.shape[0], 17, 10))

for i in range(17):
    X_reshaped[:, i, :] = X_train[:, i:i + 10]

y_reshaped = np.reshape(y_train, (y_train.shape[0], 1))

X_train = X_reshaped
y_train = y_reshaped
X_test_reshaped = np.zeros((X_test.shape[0], 17, 10))

for i in range(17):
    X_test_reshaped[:, i, :] = X_test[:, i:i + 10]

y_test_reshaped = np.reshape(y_test, (y_test.shape[0], 1))

X_test = X_test_reshaped
y_test = y_test_reshaped
print(X_train.shape)
print(y_train.shape)
X_test.dtype
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
# Output layer with sigmoid activation for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model
optimizer = optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=['accuracy'])

# Print model summary
model.summary()
import matplotlib.pyplot as plt
history=model.fit(X_train, y_train, batch_size=32, epochs=20)

plt.plot(history.history['accuracy'])
plt.title('Model Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training Loss'])
plt.show()
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score

# Get predicted probabilities
y_pred_proba = model.predict(X_test)

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall)

# Find threshold for maximum F1-score
optimal_threshold_index = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_threshold_index]

# ROC Curve
fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
desired_tpr_value = 0.9 
# Find threshold for desired TPR/FPR
desired_tpr_index = np.where(tpr >= desired_tpr_value)[0][0]
threshold_for_desired_tpr = roc_thresholds[desired_tpr_index]

# Choose the threshold that maximizes the F1-score
threshold = optimal_threshold

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='red', lw=2, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

# F1-score
thresholds = roc_thresholds  # Use the thresholds obtained from the ROC curve
f1_scores = [f1_score(y_test, (y_pred_proba >= t).astype(int)) for t in thresholds]
plt.figure(figsize=(8, 6))
plt.plot(thresholds, f1_scores, color='green', lw=2, label='F1-score')
plt.xlabel('Threshold')
plt.ylabel('F1-score')
plt.title('F1-score vs. Threshold')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
audio_file = "AUDIO\We Respond to your Dev Talk Comments - 128.mp3"
features = preprocess_audio(audio_file)
features=np.array(features)
features
feature_array = np.zeros((1, 17, 10))
for i in range(17):
    feature_array[0, i, :] = features[i:i + 10]

prediction=model.predict(feature_array)
prediction
binary_prediction = 1 if prediction <= threshold else 0
binary_prediction
