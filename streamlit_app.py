import streamlit as st
from st_audiorec import st_audiorec
import librosa
import io
import soundfile as sf
import numpy as np
import pandas as pd
import pickle
import joblib
import gdown


def feature_extractor(audio_data):
    sfo = sf.SoundFile(io.BytesIO(audio_data))
    audio, sampling_rate = librosa.load(sfo)

    features = list()
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sampling_rate))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sampling_rate))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sampling_rate))
    features.append(spectral_centroid)
    features.append(spectral_bandwidth)
    features.append(spectral_rolloff)
    mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_fft=2448)
    for el in mfcc:
        features.append(np.mean(el))

    df = pd.DataFrame([features],
                      columns=["spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
                               "mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5", "mfcc6", "mfcc7",
                               "mfcc8",
                               "mfcc9", "mfcc10", "mfcc11", "mfcc12", "mfcc13", "mfcc14", "mfcc15",
                               "mfcc16",
                               "mfcc17", "mfcc18", "mfcc19", "mfcc20"])
    return df


def scale_features(scaler_location, data):
    scaler = joblib.load(scaler_location)
    scaled_data = scaler.transform(np.array(data.iloc[:, :], dtype=float))
    return scaled_data, scaler


def select_features(feature_selector_location, data):
    feature_selector = joblib.load(feature_selector_location)
    return feature_selector.transform(data)


def predict(model_location, data):
    model = pickle.load(open(model_location, 'rb'))
    return model.predict(data)

@st.cache_resource
def load_rf_model():
    url = 'https://drive.google.com/file/d/1jrJAIflrysoOMVW6mEWd40x2OKyhWy7D/view?usp=share_link'
    output_path = 'rf_model.pkl'
    gdown.download(url, output_path, quiet=False, fuzzy=True)
    model = pickle.load(open('rf_model.pkl', 'rb'))
    return model

def predict_age(df):
    x, scaler = scale_features('age/scaler.save', df)
    encoder = joblib.load('age/label_encoder.save')
    X_new = select_features('age/feature_selector.save', x)

    knn_pred = predict('age/knn_model.pkl', X_new)
    svc_pred = predict('age/svc_model.pkl', X_new)
    rf_model = load_rf_model()
    rf_pred = rf_model.predict(X_new)

    # Applying ensemble learning
    knn_score = 0.886
    svc_score = 0.866
    rf_score = 0.691
    w = [0] * 8

    w[knn_pred[0]] += knn_score
    w[svc_pred[0]] += svc_score
    w[rf_pred[0]] += rf_score
    pred = [np.argmax(w)]
    return encoder.inverse_transform(pred)[0]

def predict_gender(df):
    x, scaler = scale_features('gender/scaler.save', df)
    encoder = joblib.load('gender/label_encoder.save')
    X_new = select_features('gender/feature_selector.save', x)
    knn_pred = predict('gender/knn_model.pkl', X_new)
    return encoder.inverse_transform(knn_pred)[0]


# DESIGN implement changes to the standard streamlit UI/UX
# --> optional, not relevant for the functionality of the component!
st.set_page_config(page_title="streamlit_audio_recorder")
# Design move app further up and remove top padding
st.markdown('''<style>.css-1egvi7u {margin-top: -3rem;}</style>''',
            unsafe_allow_html=True)
# Design change st.Audio to fixed height of 45 pixels
st.markdown('''<style>.stAudio {height: 45px;}</style>''',
            unsafe_allow_html=True)
# Design change hyperlink href link color
st.markdown('''<style>.css-v37k9u a {color: #ff4c4b;}</style>''',
            unsafe_allow_html=True)  # darkmode
st.markdown('''<style>.css-nlntq9 a {color: #ff4c4b;}</style>''',
            unsafe_allow_html=True)  # lightmode


def audiorec_demo_app():
    # TITLE and Creator information
    st.title('Detect Age from Voice')
    st.markdown('Implemented by '
                '[Rifat Monzur](https://www.linkedin.com/in/rifatmonzur/) - '
                'view project source code on '

                '[GitHub](https://github.com/rifat1234/streamlit-age-from-voice)')
    st.write('\n\n')
    st.header('Instructions:')
    st.markdown('Press \'Start Recording\' to record your voice. \n\nSay: '
                '\'Hi, I am [Your name]. Nice to meet you AI. Can you guess my age and gender?\' \n\n'
                'Then press \'Stop\'')

    # TUTORIAL: How to use STREAMLIT AUDIO RECORDER?
    # by calling this function an instance of the audio recorder is created
    # once a recording is completed, audio data will be saved to wav_audio_data
    wav_audio_data = st_audiorec()  # tadaaaa! yes, that's it! :D

    # add some spacing and informative messages
    col_info, col_space = st.columns([0.57, 0.43])
    #with col_info:
    with st.spinner("Processing... "):
        if wav_audio_data is not None:
            # display audio data as received on the Python side
            col_playback, col_space = st.columns([0.58, 0.42])
            with col_playback:
                df = feature_extractor(wav_audio_data)
                age = predict_age(df)
                gender = predict_gender(df)
                output = f"You are :blue[**{gender}**] in your :green[**{age}**]"
                st.markdown(output)


if __name__ == '__main__':
    # call main function
    audiorec_demo_app()
