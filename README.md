---
title: Instant Interview
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.23.1
app_file: app.py
pinned: true
short_description: For hosting instant interview chatbot
thumbnail: >-
  https://cdn-uploads.huggingface.co/production/uploads/67572efb860bd4d8f464793e/YgoaI2c7gb8fv53t6LXSL.jpeg
---

# Instant Interview Bot

This chatbot is hosted on [Hugging Face Spaces.](https://huggingface.co/spaces/im93/Instant_Interview/tree/main)

## Overview

This project uses generative AI to allow users to conduct an interview with a chatbot representing me. The idea here is not to pass-off the task of interviewing to AI, but rather to serve as a more immediate and accessible touch-point for recruiters, hiring managers, etc. to engage with my experience through a natural language interface while showcasing my ability to build and launch products.

## Live Demo

You can try out the live demo of the Instant Interview Bot [here.](https://sites.google.com/view/isaiahmontoya/instant-interview?)

## Model

This project uses the [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) model, which is a large language model fine-tuned for conversational tasks. The model is capable of generating human-like responses to user queries, making it suitable for use in an interview setting.

## Under the Hood

### System Message

The model is provided context and instruction via the system role. The system message is built in three steps.

1. The model is instructed to act as a chatbot representing me in the [system_prompt.txt](system_prompt.txt) document.
2. My [resume](resume.txt) is provided as context so that the model can accurately answer questions about my experience.
3. The "few-shot" method is employed to provide the model with examples of how to respond to questions via the [interview_questions_and_answers.csv](interview_questions_and_answers.csv) file. This file contains a list of common interview questions and my answers to them, which the model can use as a reference when generating responses.

### Managing Context

The model is capable of maintaining context over a conversation, but it is limited by the maximum token limit of inference client. To manage this, the oldest messages are summarized to 200 tokens using the same model, then replaced with the summary in the conversation history once the token limit is reached. This allows the model to maintain context while also ensuring that it does not exceed the token limit. Still, this method will eventually exceed the token limit, so additional logic is included to remove the oldest messages from the conversation history once the token limit is reached.

*Note: The system message is never summarized or removed from the conversation history. This is to ensure that the model always has access to the context and instruction provided.*

## Future Improvements

- Add suggested questions to the UI to help guide the conversation. Cache the answers and/or use the interview_questions_and_answers.csv file to provide answers without using inference tokens.
- Add voice input support.
- Implement a RAG system to parse through my CV and provide more accurate answers to questions without exhausting the context window.
