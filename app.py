# Program title: Simple Storytelling App (Text to Story + Audio)

import streamlit as st
from transformers import pipeline

# Set up the page
st.set_page_config(page_title="Text to Audio Story", page_icon="🦜")
st.header("Turn Your Text into an Audio Story")

# User enters text
user_text = st.text_area("Enter a prompt or scenario for your story:")

if user_text:
    # Stage 1: Text to Story
    st.text('Generating a story...')
    story_generator = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    story = story_generator(user_text)[0]['generated_text']
    st.write(story)

    # Stage 2: Story to Audio
    st.text('Generating audio data...')
    audio_generator = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    speech_output = audio_generator(story)

    # Play button
    if st.button("Play Audio"):
        audio_array = speech_output["audio"]
        sample_rate = speech_output["sampling_rate"]
        # Play audio directly using Streamlit
        st.audio(audio_array,
                 sample_rate=sample_rate)