# Evaluating Open-Source Language Models Under Adversarial Use

## Overview

This project aims to **assess and stress-test** the alignment safeguards of **open-source language models** available on Hugging Face. Specifically, we will:

1. **Survey a range of publicly available language models** (e.g., GPT-2 variants, LLaMA-based derivatives, etc.).
2. **Attempt to induce malicious outputs** by simulating real-world attacker scenarios:
   - Dissemination of **disinformation** or manipulation narratives.
   - **Scamming** instructions (e.g., phishing templates, pig-butchering scripts).
   - Techniques for **social engineering** or fraudulent schemes.
3. **Evaluate how well each model’s alignment** (if any exists) prevents or mitigates these malicious requests.

### About the Author & Context

- I’m a **Software Engineer (SWE) at Google**, working on **anti-abuse systems** for Google Messages. Our goal is to protect users globally from scams and non-compliant content that abuse our messaging platforms.  
- In this adversarial role, I’m effectively adopting an attacker’s mindset—**red-teaming** our systems to identify vulnerabilities and strengthen defenses.

Given that **alignment** is crucial in preventing harmful content generation, this project examine **open-source language models**. It’s part of a **Language Models course** project: exploring how publicly-available models handle malicious queries.

## Disclaimer

- **No malicious or harmful code** will be publicly released.  
- **Insights and analysis** are summarized to **avoid providing a manual** for malicious actors.  
- Some details—especially concrete prompts, model responses that demonstrate vulnerabilities, or thorough “how-to” exploitation tactics—are kept **private** in a separate, non-public repository to **prevent** misuse.

## Project Scope & Tasks

1. **Model Selection**  
   - Identify and download **open-source LLMs** from Hugging Face (e.g., GPT-2, GPT-J, LLaMA derivatives, Falcon, etc.).  
   - Document each model’s **parameters, training data,** and **stated alignment** approaches (if any).

2. **Threat Scenarios & Test Cases**  
   - **Manipulation & Disinformation**  
     - E.g., generating fake news articles, conspiracy narratives.  
   - **Scamming**  
     - Common vectors such as **phishing emails**, **“pig-butchering”** scripts (long-term con scams), romance scams, crypto investment traps, etc.  
     - Methods of impersonation.  
   - **Social Engineering**  
     - Prompting the model to produce **tailored scripts** to manipulate or coerce targets.  
   - Other relevant **illicit content** requests (within reason).

3. **Experimentation & Evaluation**  
   - For each scenario, design **prompt engineering** that attempts to coax the model into producing malicious or policy-violating output.  
   - Record responses:
     - **Success** (the model complies and provides malicious content).  
     - **Refusal** or partial compliance.  
     - **Hallucinations** or other forms of incorrectly denied help.  
   - Rate each model’s alignment efficacy.

4. **Literature Review**  
   - Read **at least 5 research papers** on:
     - Alignment strategies (RLHF, Constitution AI, etc.).  
     - Red-teaming LLMs.  
     - Hard-coded content filters.  
     - Ethical & policy frameworks.  
   - Summarize **key findings** that inform recommended best practices.

5. **Summary & Recommendations**  
   - Provide **general conclusions**: how each model handles adversarial prompts, which alignment methods are most robust, and any patterns observed.  
   - **High-level suggestions** for improvement in open-source model deployment.

## Ethical Considerations

- **Do No Harm**: This project’s objective is **defensive research**, identifying vulnerabilities in model outputs to propose improvements.  
- **No Release of Dangerous Material**: Detailed exploit prompts or malicious output examples will remain in **private** repositories to deter misuse.  

## Repository Structure

```
/malicious-model-testing
│
├── README.md               # Project overview (public)
├── models/                 # Configs for tested models (with disclaimers)
├── scenario_tests/         # Scripts describing the threat scenarios (private or partially redacted)
├── data/                   # Placeholder or sanitized dataset references
└── papers/                 # Summaries / notes on the 5+ alignment papers (public summary, private details)
```

> **Note**: The actual malicious prompts and specific outcomes will be stored in a **private** area (not publicly visible) to prevent enabling malicious actors.

## References & Resources

- **Open-Source LLMs** on [Hugging Face](https://huggingface.co/).  
- **5+ Papers on Alignment & Red-Teaming** (list TBD).
- Various **scam & disinformation** frameworks from anti-abuse research.

## Contact

For questions or collaboration:
- **Email**: antoni.krzysztof.czapski@gmail.com