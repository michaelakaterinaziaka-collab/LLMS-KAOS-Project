# Automation of Educational Goal Extraction and Visualization in KAOS Graphs Using LLMs

This repository contains the official implementation of the methodology presented in the paper: 
**"Automation of Educational Goal Extraction and Visualization in KAOS Graphs Using Large Language Models"** 

## Overview
Our system automates the extraction of educational goals from academic textbooks and visualizes them using the **KAOS (Goal-Oriented Requirements Engineering)** framework. By leveraging Large Language Models (LLMs), specifically Gemini-2.0-flash, we transform unstructured text into structured, hierarchical goal models.

## Key Features
- **Automated Extraction:** Identifies Main, Behavioural, and Soft Goals from educational content.
- **Formal Modeling:** Maps LLM outputs to KAOS-compliant structures (AND/OR refinements, conflicts).
- **Visualization:** Generates high-quality goal graphs using Graphviz.
- **Explainability:** Provides a "white-box" layer to audit AI-generated curriculum structures.

## How to Run the project
1. **Clone or download this repository**
2. **Set up your API Key**: Open the file extract_goals_gemini_translated.py and locate line 87. Replace "API_KEY" with your actual Google Gemini API Key. Open the file AI_evaluation_translated.py and locate line 6. Replace "API_KEY with your actual OPENAI API Key.
3. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
4. **Execute the scripts**:
   -To extract goals: extract_goals_gemini_translated.py
   -To generate graphs: generate_gore_graphs_translated.py
   -To evaluate using another LLM as a judge: AI_evaluation_translated.py

## Note on Language & Reproducibility
The original experiments were conducted using Greek educational material from the **"Kallipos"** dataset. 
- For the purpose of this repository and consistency with the submitted paper, the prompts inside the Python scripts have been **translated into English**. 
- Both the English and the original Greek versions have been verified to produce equivalent structural results (JSON output).
- The system is designed to handle Greek input text even with English instructions.
