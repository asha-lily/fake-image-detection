# Fake Image Detection: Research Notes

This project is a work in progress. This page documents the initial research process. No code has been written yet.

## Table of Contents
- [Introduction](#introduction)
    - [Motivation](#motivation)
    - [Aims](#aims)
- [How are synthetic images generated?](#how-are-synthetic-images-generated)
- [How are synthetic images detected?](#how-are-synthetic-images-detected)
- [Literature Review](#literature-review)
    - [Summary](#summary)
- [Experiment Ideas](#experiment-ideas)


## Introduction

### Motivation

The motivation for this research project comes from the increasing prevalence of AI-generated images in various aspects of everyday life and my concerns about the potential impacts of society being able to easily generate and share such content. To name just a few of these concerns:

- Online scams where the buyer relies on an image
- Fabricated political / high-profile events (and other types of misinformation)
- Non-consensual intimate imagery

### Aims

The technology used to generate images, video, audio etc is advancing faster than our ability to reliably detect synthetic content. As the European Parliament notes in their [2025 briefing](https://www.europarl.europa.eu/RegData/etudes/BRIE/2025/775855/EPRS_BRI%282025%29775855_EN.pdf) on 'Children and deepfakes': '*no single robust solution currently exists to detect and reduce the spread of harmful AI-generated content.*'

I'd like to learn more about both how this content - specifically images - is generated and how we can detect it. 

Once I have an understanding of the current state of research in this area, I plan to run experiments of my own, fine-tuning open-source models to classify real vs synthetic images. Given time and computational resource constraints, my aim won't be to produce the best model possible, but rather to see what can be achieved with models such as CNNs and transformers.

I'm also interested in using class activation map methods to visualise image artifacts that models learn in order to distinguish real from synthetic. 


## How are synthetic images generated?

- GANs
    - commonly used for face synthesis, e.g e.g StyleGAN
    - also used for face morphing, e.g for generating synthetic identities
    - can be used to generate speech & synchronise lip movements with audio in videos, e.g Wav2Lip
- Autoencoders
    - used for face swapping
- Diffusion models


## How are synthetic images detected?

- Traditional techniques
    - digital "forensics", e.g looking at patterns in noise
    - these techniques can be bypassed by modern generation models
- CNNs
- Transformers
- Autoencoders
- RNNs & LSTMs
    - can detect inconsistencies across video frames
- Watermarking
    - e.g Google SynthID
- C2PA
    - an industry-wide standard that uses cryptographically-signed metadata to provide secure & verifiable records of a media file's origin and changes


## Literature Review

While the most recent research is of interest, I have also sourced some older papers (2023) in order to understand how research has progressed over the past few years.Another purpose of this literature review is to source data that I can use in my own experiments.

[AI-Generated Image Detection: Baraheem et al.](https://www.mdpi.com/2313-433X/9/10/199) (Sept 2023)
- Creates a framework to detect GAN-generated images using transfer learning on pre-trained CNN classifiers
- Built the RSI dataset: 48000 images created by 12 GAN architectures
- Fine-tuned EfficientNetB4 on the RSI dataset
- Incorporated 4 different CAM techniques for explainability

[CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images](https://ieeexplore.ieee.org/document/10409290) (Jan 2024)
- Introduces the CIFAKE dataset: synthetic equivalents of CIFAR-10 generated using stable diffusion
- Trains a CNN for classifying real vs AI-generated images
- Implements Grad CAM to highlight regions influencing the model's decisions. These heatmaps reveal that the model focuses on subtle imperfections, often in the background, to distinguish real vs synthetic
 
In my opinion the images in the CIFAKE dataset look clearly AI-generated, which is unsuprising given that this paper is a couple of years old and diffusion models have surpassed GANs as state-of-the-art.

[Towards universal fake image detectors that generalize across generative models](https://arxiv.org/pdf/2302.10174) ('UniversalFakeDetect') (April 2024)
- Highlighted that existing fake image detectors struggle to generalise to images from different generative models when trained on GAN-generated images
- To address this, the authors propose constructing a feature space using CLIP:ViT, e.g using nearest neighbour search to classify real vs fake

[AI-Generated Image Detection: An Empirical Study and Future Research Directions](https://arxiv.org/pdf/2511.02791) (Nov 2025)
Highlights the following issues across AI-generated image detection research:
- The limitations of forensic methods
- The use of non-standardised benchmarks with GAN- or diffusion-generated images
- Inconsistent training protocols
- Limited evaluation metrics that fail to capture generalisation & explainability

[FakeXplained dataset](https://arxiv.org/html/2506.07045v1#S3) (June 2025)
- '*we aim to train MLLMs not only to detect AI-generated images but also to articulate why they are fake in a reliable and human-understandable manner. This necessitates a dataset that supports both visual grounding and textual reasoning.*'
- Produced ~9000 AI-generated images annotated with bounding boxes & descriptive captions highlighting synthesis artifacts

[ThinkFake: Reasoning in Multimodal LLMs for AI-generated Image Detection](https://arxiv.org/pdf/2509.19841)
- Highlights that directly prompting MLLMs (e.g 'explain what the artifacts are') to generate textual explanations often results in hallucinations or overthinking, leading to inaccuracte outcomes or refusal to respond
- Researchers are employing fine-tuning approaches such as LoRA or DPO to overcome these limitations; these methods tend to memorise training patterns, which is then addressed using GRPO to enhance the model's ability to 'think'

[Towards Explainable Fake Image Detection with Multi-Modal Large Language Models](https://arxiv.org/pdf/2504.14245) (Nov 2025)
- Designed 6 specialised prompts, each targeting a distinct visual or logical aspect of an image. A majority vote is taken from across the 6 results to provide the classification.
- Created a dataset of 2000 images produced by various methods including diffusion & GAN
- Benchmarked 4 major multi-modal LLMs against other detectors

In this study I found it interesting that the models generally rejected (i.e refused to provide a response for) fewer images when the word `fake` in the prompt was replaced with `generated`. This highlights the sensitivity of LLMs to the precise wording of the prompt and the potential unreliability that results from this combined with the fact that explainability can be impacted by hallucinations (which is also highlighted in [ThinkFake](https://arxiv.org/pdf/2509.19841)).

The [FakeExplained](https://arxiv.org/pdf/2504.14245) paper highlights the same issues, but attempts to address them in different ways. Like FakeExplained, the [AIGI-Holmes dataset](https://huggingface.co/datasets/zzy0123/AIGI-Holmes-Dataset) also labels images with bounding boxes and descriptions. Unfortunately, due to the lack of standardised benchmarks (as highlighted [here](https://arxiv.org/pdf/2511.02791)), the performance of these approaches can't be compared side-by-side.

This study frames the limitations of MLLMs in a broader context: '*while MLLMs show promise in detecting AI-generated images, challenges remain in interpretability and alignment with human perception*...*ethically, ensuring transparency and accountability in detection models is critical, especially in sensitive areas like forensics and law enforcement.*'

### Summary
- Over the past few years, AI-generated image detection techniques have evolved from fine-tuning CNNs, to transformers and more recently, multi-modal LLMs.
- Data was initially generated using GANs, but  diffusion models have become increasingly popular (I haven't yet explored the details of why this is)
- CAM-based explainability techniques are often employed to identify areas of the image that the model focuses on / detects artifacts in
- MLLMs, with their reasoning abilities, show some promise, but also present new challenges such as hallucinations


## Experiment Ideas

This research has given me some ideas to build upon my original plan of fine-tuning a model to classify real vs synthetic images. 

My initial focus will be on fine-tuning a vision transformer. I like the idea implemented in [UniversalFakeDetect](https://arxiv.org/pdf/2302.10174): training on GAN-generated images and testing on diffusion-generated images in order to test the model's ability to generalise to images from a different model type. 

Beyond this, I'd be interested to compare my model's performance to that of an MLLM, experimenting with prompt engineering techniques and potentially fine-tuning.

Whether any of this is feasible depends on what data I can source. The next section will explore what suitable real & AI-generated data is publicly available.

