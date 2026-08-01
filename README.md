# Fake Image Detection: Research Notes

This project is a work in progress. 

This page documents the initial research process.

The process of building the dataset is documented in `notebooks/dataset_exploration.ipynb`.

The actual training process is initiated from `train.py`, which imports from python files in `fake_image_detection`.

## Table of Contents
- [Introduction](#introduction)
    - [Motivation](#motivation)
    - [Aims](#aims)
- [How are synthetic images generated?](#how-are-synthetic-images-generated)
- [How are synthetic images detected?](#how-are-synthetic-images-detected)
- [Literature Review](#literature-review)
    - [Summary](#summary)
- [Experiment Ideas](#experiment-ideas)
- [Dataset](#dataset)
    - [Synthetic Images](#synthetic-images)
    - [Real Images](#real-images)
    - [Image Augmentations](#image-augmentations)
    - [The effect of image quality on overfitting in synthetic image detection](#the-effect-of-image-quality-on-overfitting-in-synthetic-image-detection)
- [Choosing a pre-trained vision transformer](#choosing-a-pre-trained-vision-transformer)
    - [ViT trained on ImageNet](#vit-trained-on-image-net)
    - [ViT trained on CLIP](#vit-trained-on-clip)
    - [Patch Sizes](#patch-sizes)


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

Given enough time I'd be interested in using class activation map methods to visualise image artifacts that models learn in order to distinguish real from synthetic. 


## How are synthetic images generated?

**GANs**
- Consist of a generator network that creates synthetic content, alongside a discriminator which tries to distnguish real vs synthetic. The two networks are trained in an adversarial process.
- Commonly used for face synthesis, e.g e.g StyleGAN
- Also used for face morphing, e.g for generating synthetic identities
- Can be used to generate speech & synchronise lip movements with audio in videos, e.g Wav2Lip

**Autoencoders**
- An encoder embeds the image; a decoder reconstructs the image from the embedding
- Face swapping can be achieved by exchanging the encoded features between different images
- *Variational* autoencoders are a distinct type of autoencoder which encode a continuous, probabilistic representation of the latent space (as opposed to a fixed, discrete representation). By sampling from this distribution they can generate new data similar to the original training data.

**Diffusion models**
- Trained by iteratively adding noise to an image and trying to recreate the original image from the noise
- Faster to train than GANs and don't require as much data, but slower at inference

### Timeline

- 2014 - 2020: VAEs & GANs (e.g StyleGAN, BigGAN)
- 2020 - 2021: Pixel diffusion (e.g DDPM, guided diffusion)
- 2022 - 2023: Text-to-image latent diffusion (DALL-E 2/3, Imagen, Midjourney)
- 2024: Flow matching + DiT backbones (e.g SD3, FLUX.1)
- 2025 - 2026: Native multi-modal LLM generation

There isn't a definitive source for which synthetic-image-generation models are the 'best' at the moment. However, a few that I found often appeared near the top of the rankings include:

- FLUX.2 Pro: known for its high quality
- Midjourney V8.1: known for its artistic style
- Stable Diffusion 3.5: open-source


## How are synthetic images detected?

- Traditional techniques
    - digital "forensics", e.g looking at patterns in noise
    - modern generators don't leave behind the artifacts that these techniques rely on
- CNNs
    - Notably, the 2020 Wang et al paper (see [literature review paper 1](#paper-1)) found that systematic flaws in CNN-generated images could be detected by a CNN classifier.
    - CNNs remain competitive as synthetic image detectors  
- Frequency / spectral methods
    - See [literature review paper 10](#paper-10), which detects spectral artifacts created by GANs
    - Diffusion models leave weaker spectral artifacts than GANs, so this method is less effective at detection diffusion-generated images.
- Transformers
    - E.g fine-tuned vision transformer classifiers
    - Some studies have found that transformers outperform CNNs on this task, which may be due to their ability to learn global features which are more likely to survive image transformations than the local features that CNNs learn.
- Autoencoders
    - An image is passed through the autoencoder of a latent diffusion model
- Multi-modal LLMs (MLLMs)
    - Detects semantic signals like impossible geometry, lighting inconsistencies (e.g between a subject and the background) etc, however these inconsistencies are becoming less common as generation methods improve

#### Watermarking & C2PA

Some model providers such as Google add invisible watermarks to their images to enable end-users to identify the image as AI-generated (only certain models / platforms such as Google's Gemini can detect the watermark). While Google's SynthID watermark was designed to be robust to image transformations / manipulations such as compression and filtering, there is some evidence that the watermark can be removed.

C2PA is another approach to labelling AI-generated content. It uses cryptographically-signed metadata to provide secure & verifiable records of a media file's origin and changes. Currently, C2PA metadata gets stripped when downloading an image from a social media site or simply taking a screenshot of an image.

In August 2026 the EU AI Act will mandate that AI-generated image, audio and text must be tagged as AI-generated, using both a machine-readable watermark and a human-readable label.


## Literature Review

While the most recent research is of interest, I have also sourced some older papers (2023) in order to understand how research has progressed over the past few years. Another purpose of this literature review is to source data that I can use in my own experiments.

#### Paper 1: [CNN-generated images are surprisingly easy to spot... for now](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_CNN-Generated_Images_Are_Surprisingly_Easy_to_Spot..._for_Now_CVPR_2020_paper.pdf)
- Collected a dataset of fake images generated by 11 different CNN-based image generation models, including ProGAN, StyleGAN etc.
- Trained a CNN classifier (ResNet-50) on ProGAN images
- Applied 'careful' pre- & post-processing, along with data augmentation
- The trained model generalisee 'suprisingly well' to images generated by models other than ProGAN
- Concluded that CNN-generated images share common systematic flaws which prevent them from synthesising realistic images

#### Paper 2: [AI-Generated Image Detection: Baraheem et al.](https://www.mdpi.com/2313-433X/9/10/199) (Sept 2023)
- Creates a framework to detect GAN-generated images using transfer learning on pre-trained CNN classifiers
- Built the RSI dataset: 48000 images created by 12 GAN architectures
- Fine-tuned EfficientNetB4 on the RSI dataset
- Incorporated 4 different CAM techniques for explainability

#### Paper 3: [CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images](https://ieeexplore.ieee.org/document/10409290) (Jan 2024)
- Introduces the CIFAKE dataset: synthetic equivalents of CIFAR-10 generated using stable diffusion
- Trains a CNN for classifying real vs AI-generated images
- Implements Grad CAM to highlight regions influencing the model's decisions. These heatmaps reveal that the model focuses on subtle imperfections, often in the background, to distinguish real vs synthetic
 
In my opinion the images in the CIFAKE dataset look clearly AI-generated, which is unsuprising given that this paper is a couple of years old and diffusion models have surpassed GANs as state-of-the-art.

#### Paper 4: [Towards universal fake image detectors that generalize across generative models](https://arxiv.org/pdf/2302.10174) ('UniversalFakeDetect') (April 2024)
- Highlighted that existing fake image detectors struggle to generalise to images from different generative models when trained on GAN-generated images
- To address this, the authors propose constructing a feature space using CLIP:ViT, e.g using nearest neighbour search to classify real vs fake

#### Paper 5: [AI-Generated Image Detection: An Empirical Study and Future Research Directions](https://arxiv.org/pdf/2511.02791) (Nov 2025)

Highlights the following issues across AI-generated image detection research:
- The limitations of forensic methods
- The use of non-standardised benchmarks with GAN- or diffusion-generated images
- Inconsistent training protocols
- Limited evaluation metrics that fail to capture generalisation & explainability

#### Paper 6: [FakeXplained dataset](https://arxiv.org/html/2506.07045v1#S3) (June 2025)
- '*we aim to train MLLMs not only to detect AI-generated images but also to articulate why they are fake in a reliable and human-understandable manner. This necessitates a dataset that supports both visual grounding and textual reasoning.*'
- Produced ~9000 AI-generated images annotated with bounding boxes & descriptive captions highlighting synthesis artifacts

#### Paper 7: [ThinkFake: Reasoning in Multimodal LLMs for AI-generated Image Detection](https://arxiv.org/pdf/2509.19841)
- Highlights that directly prompting MLLMs (e.g '*explain what the artifacts are*') to generate textual explanations often results in hallucinations or overthinking, leading to inaccuracte outcomes or refusal to respond
- Researchers are employing fine-tuning approaches such as LoRA or DPO to overcome these limitations; these methods tend to memorise training patterns, which is then addressed using GRPO to enhance the model's ability to 'think'

#### Paper 8: [Towards Explainable Fake Image Detection with Multi-Modal Large Language Models](https://arxiv.org/pdf/2504.14245) (Nov 2025)
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

## Dataset

### Synthetic Images

My research into synthetic datasets is summarised in the table below. Note that I looked specifically for GAN-generated and diffusion-generated images.

| Dataset Name | Model Type | Year of creation | Good enough quality to use? | Num real | Description of real images | Num synthetic | Description of synthetic images | Licence |
|-----|-----|-----|-----|-----|----------|-----|----------|-------|
| [diffusion_datasets](https://github.com/WisconsinAIVision/UniversalFakeDetect) | Diffusion | 2020 | No | 1000 | imagenet | 9000 | 1000 images from 9 different models | MIT (no restrictions)
| [progan](https://github.com/WisconsinAIVision/UniversalFakeDetect) | GAN | 2020 | No | 4200 | 21 classes (objects & animals); 201 of each | 4200 | 21 classes (objects and animals); 201 of each | MIT (no restrictions) |
| [dragon_train_xs](https://huggingface.co/datasets/lesc-unifi/dragon/tree/main) | Diffusion | 2024 - 2025 | Maybe | 0 | | 250 |  25 different models; only 10 images from each (same 10 prompts given to each model, so images are very similar) | Creative commons (fine for commercial & private use) |
| [AIS-4SD](https://zenodo.org/records/15131117) | Diffusion | 2025 | Only 500 faces are usable (StableDiffusion-3-faces-20250203-1545) | 0 | | 4000 | 4 different models; 1000 images from each. 500 of people & 500 of other generic things | MIT |
| [SFHQ-T2I](https://www.kaggle.com/datasets/selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models/data) | Diffusion | 2023 / 2024 | Yes | 0 | | 1700 |  All human faces. Produced by 2 different models. | MIT |
| [SFHQ_part1](https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-1) | GAN | 2022 / 2023 | Yes | 0 | | 550 | All human faces | Creative commons |
| [CocoGlide](https://arxiv.org/pdf/2212.10957) | Diffusion | 2022 | Maybe | 512 | | 512 | The synthetic images are very similar to the real ones - model just used for in-painting, not generation | Can’t find the original source! |

From across AIS-4SD & SFHQ-T2I we have 2200 diffusion-generated images of human faces, so I'm restricted to focusing my experiments on human faces. Unfortunately the real images I’ve found so far are not of human faces, so I need to look for some of these. 

One thing I need to look into is the dataset sizes that researchers have used in similar experiments. Given how difficult it's been to find the data summarised in the table, we'll initially use the ~2000 diffusion-generated images and look for 2000 real images of human faces to use alongside them.

I could of course generate my own dataset; I will consider this in future work, but in the interest of time and resources I will use publicly avaiable data for now.

Another idea for future work is to evaluate performance on test sets from different models, .eg
- Baseline diffusion-generated images similar to the training data (human faces)
- Diffusion-generated images that are not faces
- GAN-generated images: we have 550 human faces from SFHQ_part1


### Real Images

| Dataset Name | Year of creation | Good enough to use? | Num real | Description | Licence |
| ----- | ----- | ----- | ----- | ----- | ----- |
| celeba | 2018 | No | 200,000 | Poor quality | N/A |
| FFHQ | 2022 | Yes | 3000 | Produced by NVIDIA as part of the original StyleGAN paper | Creative commons: You can use, redistribute, and adapt it for non-commercial purposes, as long as you (a) give appropriate credit by citing our paper, (b) indicate any changes that you've made, and (c) distribute any derivative works under the same license. |

### Image Augmentations

[This paper](https://www.peren.gouv.fr/en/perenlab/2025-02-11_ai_summit/#lenjeu-interroger-les-d%C3%A9tecteurs-%C3%A0-l%C3%A9tat-de-lart-%C3%A0-bon-escient) from the French government focuses on the use of AI-generated content on social media, and the difficulty of detecting it. They apply transformations such as JPEG compression, addition of text, aesthetic filters and resizing, with the aim of imitating the progressive alteration of images as they are shared across social media. They highlight that manipulating synthetic images in ways such as these degrade the images and mask flaws related to their generation, *‘making it easier to deceive users and also impairing the capabilities of detection systems.’*

Image augmentations are especially important to prevent patterns in image quality, composition etc - which differ between the real & synthetic datasets - being learned by the model. With the datasets I’m using, composition is something I’m concerned about because the real images are cropped quite close to the faces, and they tend to look head-on at the camera, whereas in the synthetic images the pose varies more and the position of and amount of background around the person varies. Therefore we should consider:
- Applying random crops to both sets of images
- To crop the synthetic images closer to the faces, perhaps we could use a face detection model to get a bounding box around the face and crop slightly outside of this?
- Rotate the images

To account for differences in the quality of the real & synthetic images, I will consider the following types of augmentations:
- Adding blur / noise / jpeg compression to the diffusion-generated images to make the look lower quality (i.e more similar to the real images)
- Vary image resolution

There’s a huge amount of exploration I *could* do into the diversity of the real vs synthetic datasets in order to identify things that the model could learn to detect as a proxy for real vs synthetic. For example, skin tone, lighting, accessories such as glasses or hats, background (e.g inside vs outside), hair colour etc. We should apply transformations that vary these features as much as possible. Since we can't realistically account for all of these, we must acknowledge them as potential limitations in the model’s ability to learn features that actually distinguish real vs synthetic.

### The effect of image quality on overfitting in synthetic image detection
 
I did some research into the effect of image quality on overfitting in synthetic image detection.

#### Paper 9: [A New Approach to Improve Learning-based Deepfake Detection](https://arxiv.org/pdf/2203.11807) (March 2022)
- Addresses models overfitting to quality differences rather than semantic features.
- '*Training with augmentations on the same dataset remarkably improves performance on nearly all kinds of processed data even with intense severity, including JPEG compression, Gaussian noise, Gaussian blur, and Gamma correction*'

#### Paper 10: [Any-Resolution AI-Generated Image Detection by Spectral Learning](https://openaccess.thecvf.com/content/CVPR2025/papers/Karageorgiou_Any-Resolution_AI-Generated_Image_Detection_by_Spectral_Learning_CVPR_2025_paper.pdf) (March 2025)
- Performance drops in all cases when augmentations are removed, highlighting their value. 

#### Paper 11: [Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets](https://arxiv.org/html/2509.21864v1) (Sept 2025)
- '*Strong biases exist in existing benchmarks toward JPEG compression (real images: compressed, fake images: uncompressed). Many detectors inadvertently learn to detect JPEG artifacts rather than generation artifacts*'.

#### Paper 12: [Generalized Design Choices for Deepfake Detectors](https://arxiv.org/html/2511.21507) (Nov 2025)
- '*While data augmentation is critical for robust detection, excessively strong augmentations may be counterproductive; augmentations that closely mimic realistic post-processing operations encountered in-the-wild provide more consistent improvements*'
- Found that introducing repeated JPEG compression passes during training improves generalization capabilities

In summary, the research validates my concerns about image quality and strongly recommends applying augmentations such as JPEG compression.

## Choosing a pre-trained vision transformer

Most open-source ViTs were pre-trained on ImageNet or CLIP. There is also the Laion dataset, which was used to train stable diffusion among other text-to-image models. It comes in 2 versions: one trained on 400 million images and the other trained on 5 billion.

### ViT trained on ImageNet
- The original release of ImageNet contained [3 ‘people’ categories](https://www.image-net.org/update-sep-17-2019.php): scuba diver, bridegroom & baseball player, out of a total of 1000 categories. Other categories may contain people, but they are not the main subject of the image.
- In 2019, for privacy reasons all human faces in ImageNet were blurred

### ViT trained on CLIP
- [While OpenAI has never explicitly specified or shared the data used to train the original CLIP model, the CLIP paper mentions that the model was trained on 400 million image-text pairs collected from the Internet](https://voxel51.com/blog/a-history-of-clip-model-training-data-advances).
- Presumably this data contained some human faces
- Given that the faces in ImageNet were blurred but those used to train CLIP were not, perhaps CLIP is a better choice. Comparing the performance of the two is another experiment we could do, but to start with let’s use CLIP.
- We can look on hugging face for open-source vision transformers that have been pre-trained on CLIP
- Another choice to make is patch size

### Patch sizes
- The standard patch size for vision transformers is 16 x 16
- Smaller patch sizes suit tasks that involve detecting small objects or high-resolution details, however their use also reduces computational efficiency. We'll start with a patch size of 16 x 16 and review whether there are any issues with computational efficiency.
- Image resolution should be divisible by patch size!

Next, see `notebooks/dataset_exploration.ipynb`, which documents building the dataset.

### Future Ideas
- does balancing the classes improve performance
    - stratified sampling
- K-fold cross-validation? Since we don't have a huge amount of data
- performance metrics
- batch size (depends on GPU memory capacity)
- effect of num steps on performance


# Questions / To Do
- what is pixel space diffusion vs latent diffusion?
- review 'How are synthetic images generated?' section
- NITRE robust AI-generated image detection in the wild
- review 'How are synthetic images detected?' - provide details for each bullet point?