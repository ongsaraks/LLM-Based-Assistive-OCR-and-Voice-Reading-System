# LLM-Based Assistive OCR and Voice Reading System

This project implements an assistive AI system designed to read printed text aloud for blind and visually impaired users using OCR and large language models.

## Problem Statement
Blind and visually impaired users face difficulties accessing printed information such as product labels, medicine instructions, and expiry dates.  
Existing OCR tools often output raw text that is unclear, incomplete, or unsuitable for voice-based understanding.

## What I Did
- Designed an end-to-end OCR-to-voice system for real-world label reading  
- Integrated an OCR API capable of extracting Thai and English text  
- Applied prompt engineering and few-shot prompting to interpret OCR output clearly  
- Implemented AI-based content interpretation optimized for text-to-speech  
- Built automatic handling for Thai and English mixed-language labels  
- Developed a backend service to preprocess images and generate spoken output  

## Technologies Used
- Typhoon OCR API  
- Typhoon LLM API  
- Python  
- Flask  
- OpenCV  
- Text-to-Speech (gTTS)  
- IoT camera device (ESP32 or equivalent)

## Outcome
- Reads product labels, ingredients, warnings, and expiry dates aloud  
- Supports both Thai and English, including mixed-language text  
- Produces clear, concise, and complete voice descriptions suitable for accessibility use  

## Notes
This project was developed in an assistive technology context.  
Accuracy, completeness, and ethical handling of sensitive information are critical for real-world deployment.
