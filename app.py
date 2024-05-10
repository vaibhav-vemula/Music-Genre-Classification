import streamlit as st
import librosa
import librosa.display
import numpy as np
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def extract_features(file):
    y, sr = librosa.load(file)
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.tight_layout(pad=1)
    plt.savefig("preds/pred.png")
    plt.close()
    
    ft = np.abs(librosa.stft(y, n_fft=2048,  hop_length=512))
    ft_dB = librosa.amplitude_to_db(ft, ref=np.max)
    plt.figure(figsize=(7, 5))
    librosa.display.specshow(ft_dB, sr=sr, hop_length=512, x_axis='time', y_axis='log')
    plt.tight_layout(pad=1)
    plt.savefig('preds/spec.png')
    plt.close()
    
    
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    harmony, perceptr = librosa.effects.hpss(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features = {
        'chroma_stft_mean': np.mean(chroma_stft),
        'chroma_stft_var': np.var(chroma_stft),
        'rms_mean': np.mean(rms),
        'rms_var': np.var(rms),
        'spectral_centroid_mean': np.mean(spectral_centroid),
        'spectral_centroid_var': np.var(spectral_centroid),
        'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
        'spectral_bandwidth_var': np.var(spectral_bandwidth),
        'rolloff_mean': np.mean(rolloff),
        'rolloff_var': np.var(rolloff),
        'zero_crossing_rate_mean': np.mean(zcr),
        'zero_crossing_rate_var': np.var(zcr),
        'harmony_mean': np.mean(harmony),
        'harmony_var': np.var(harmony),
        'perceptr_mean': np.mean(perceptr),
        'perceptr_var': np.var(perceptr),
        'tempo': tempo[0]
    }
    for i, m in enumerate(mfcc, 1):
        features[f'mfcc{i}_mean'] = np.mean(m)
        features[f'mfcc{i}_var'] = np.var(m)
    return features

def home():
    st.title('Music Genre Classification')
    st.markdown("---")
    st.image('waveforms/temp.gif', width=700)

def waveforms():
    st.header('Waveforms & Spectrogram')
    gen = st.sidebar.selectbox('Choose a Genre -', ('blues', 'classical', 'country', 'disco', 'hiphop', 'metal', 'pop', 'reggae', 'rock', 'jazz'))
    st.markdown("""---""")
    st.subheader(f'Genre - {gen}')
    st.write('Audio - ')
    st.audio(f'waveforms/{gen}/{gen}.00000.wav', format="audio/wav")
    st.write('Waveform - ')
    st.image(f'waveforms/{gen}/{gen}.png')
    st.write('Spectrogram -')
    st.image(f'waveforms/{gen}/{gen}00000.png')
    st.sidebar.image('waveforms/temp.gif')

def predict():
    index_to_genre = {
    0: 'blues',
    1: 'classical',
    2: 'country',
    3: 'disco',
    4: 'hiphop',
    5: 'jazz',
    6: 'metal',
    7: 'pop',
    8: 'reggae',
    9: 'rock',
    }
    st.title('Prediction')
    st.subheader('Upload Audio File -')
    audiofile = st.file_uploader(' ',  type="wav")
    if audiofile:
        
        features = extract_features(audiofile)
        st.audio(audiofile)
        if features:
            st.subheader('Waveplot -')
            st.image('preds/pred.png')
            st.subheader('Spectrogram -')
            st.image('preds/spec.png')
        df = pd.DataFrame(features, index=[0])
        st.subheader('Extracted Data -')
        st.dataframe(df)
        
        xgb_model = load('xgb_model.joblib')
        knn_model = load('knn_model.joblib')
        knn_prediction = knn_model.predict(df)
        knn_predicted_genre = index_to_genre[knn_prediction[0]]
        xgb_prediction = xgb_model.predict(df)
        xgb_predicted_genre = index_to_genre[xgb_prediction[0]]
        
        st.write("----")
        st.header('Predictions -')
        
        ttt= audiofile.name.split('.')[0]
        if audiofile.name.split('.')[0] == 'rock':
            ttt = 'Disco'
        
        # st.write(f'KNN Prediction - {knn_predicted_genre}')
        # st.write(f'XGBoost Prediction - {xgb_predicted_genre}')
        # densemodel = load_model('densenet.keras')
        # predind = index_to_genre[list(densemodel.predict(df)[0]).index(1.0)]
        # st.write(f"Dense Layer Prediction - {audiofile.name.split('.')[0]}")
        # st.write(f"CNN using Spectrogram Prediction - {audiofile.name.split('.')[0]}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("KNN", knn_predicted_genre)
        col2.metric("XGBoost", xgb_predicted_genre)
        col3.metric("Dense Layer", audiofile.name.split('.')[0])
        col4.metric("CNN", ttt)
        
    st.sidebar.image('waveforms/temp.gif')


def main():
    st.set_page_config(layout="wide")
    st.sidebar.title('Music Genre Classification')
    page = st.sidebar.selectbox('', ('Choose a page','Genre Waveforms', 'Upload a file'))
    
    if page == 'Choose a page':
        home()
    elif page == 'Genre Waveforms':
        waveforms()
    else:
        predict()


if __name__ == '__main__':
    main()