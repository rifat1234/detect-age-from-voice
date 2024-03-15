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

@st.cache_resource
def load_rf_model():
    url = 'https://drive.google.com/file/d/1jrJAIflrysoOMVW6mEWd40x2OKyhWy7D/view?usp=share_link'
    output_path = 'rf_model.pkl'
    gdown.download(url, output_path, quiet=False, fuzzy=True)
    model = pickle.load(open('rf_model.pkl', 'rb'))
    return model

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
                '\'Hi, I am [Your full name]. Nice to meet you AI. Can you guess my age from my voice?\' \n\n'
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
                sfo = sf.SoundFile(io.BytesIO(wav_audio_data))
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

                df = pd.DataFrame([features], columns=["spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
                                                       "mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5", "mfcc6", "mfcc7",
                                                       "mfcc8",
                                                       "mfcc9", "mfcc10", "mfcc11", "mfcc12", "mfcc13", "mfcc14", "mfcc15",
                                                       "mfcc16",
                                                       "mfcc17", "mfcc18", "mfcc19", "mfcc20"])

                def scale_features(data):
                    scaler = joblib.load('scaler.save')
                    scaled_data = scaler.transform(np.array(data.iloc[:, :], dtype=float))
                    # with data.iloc[:, 0:-1] we don't consider the label column

                    return scaled_data, scaler

                x, scaler = scale_features(df)
                encoder = joblib.load('label_encoder.save')

                f_selector = joblib.load('feature_selector.save')
                X_new = f_selector.transform(x)

                knn_model = pickle.load(open('knn_model.pkl', 'rb'))
                knn_pred = knn_model.predict(X_new)
                svc_model = pickle.load(open('svc_model.pkl', 'rb'))
                svc_pred = svc_model.predict(X_new)
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

                output = f"You are in your  :green[**{encoder.inverse_transform(pred)[0]}**]"
                st.markdown(output)


if __name__ == '__main__':
    # call main function
    audiorec_demo_app()