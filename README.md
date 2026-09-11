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

The technology used to generate images, video, audio etc is advancing faster than our ability to reliably detect synthetic content. As the European Parliament notes in their 2025 briefing on 'Children and deepfakes': '*no single robust solution currently exists to detect and reduce the spread of harmful AI-generated content.*'[^eu-parliament]

I'd like to learn more about how synthetic images are generated and how we can detect them. 

Once I have an understanding of the current state of research in this area, I plan to run experiments of my own. Given time and computational resource constraints, my aim won't be to produce the best model possible, but rather to see what can be achieved by fine-tuning models on ~1000s of images.

Given enough time I'd be interested in using techniques such as class activation map methods to visualise image artifacts that models learn in order to distinguish real from synthetic. 


## How are synthetic images generated?

The main architectures and methods are outlined in this section.

#### Autoregressive
- This is a method which treats images as a sequence of pixels or tokens, and predicts them one at a time based on the previous ones. The underlying architecture can be convolutional or transformer-based.
- This was found to be very slow when operating at the pixel-level, but was improved by encoding each image patch as a token from a vocabulary of visual patterns
- An example of an autoregressive image generation model is DALL-E1, released in 2021. This has a transformer architecture which predicts the image tokens then passes them to a VQ-VAE decoder to produce pixels. We won't explore the details of the architecture here, but the following blog posts explain them very well: [^dalle1-1][^dalle1-2].
- Ultimately autoregressive models were overtaken by diffusion models, which are more efficient and better at rendering the image. For example DALL-E1's successor DALL-E2 replaced autoregressive image generation with diffusion. However, we'll see at the end of this section that autoregression has made a come back in image generation with the use of LLMs.

#### Autoencoders
- This is a model architecture consisting of an encoder and a decoder.
- An encoder embeds the image; a decoder reconstructs the image from the embedding. These are convolutional neural networks.
- Face swapping can be achieved by exchanging the encoded features between different images
- *Variational* autoencoders (VAEs) are a distinct type of autoencoder. While a basic autoencoder encodes an image to the same set of features every time (i.e it's deterministic), a variational autoencoder encodes a probaility distribution for each feature. Regularisation smooths this latent space, therefore by sampling from it they can generate new data similar to the original training data. 
- While autoencoders were never widely used as images generators due to them generating blurry images, they became an important component in latent diffusion models and many native multi-modal LLMs.

#### GANs
- This architecture consists of a generator network that creates synthetic content, alongside a discriminator which tries to distinguish real vs synthetic. The two networks are trained in an adversarial process.
- Early GANs consisted of fully-connected layers, while later GANs such as ProGAN and StyleGAN consist of convolutional layers.
- Commonly used for face synthesis, e.g StyleGAN. Also used for face morphing, e.g for generating synthetic identities
- Can be used to synchronise lip movements with audio in videos, e.g Wav2Lip

#### Diffusion models
- Diffusion is a method in which noise is iteratively added to an image and a model predicts the added noise. At inference time this process is reversed to produce an image from noise, conditioned on a text prompt.
- The original diffusion models used for image generation had U-Net architectures. Like autoencoders, these are convolutional neural networks consisting of an encoder and decoder, however the output of a U-Net is not the same as the input. The U-Net was designed for image segmentation, i.e outputting a 'mask' the same size as the input image but with each pixel labelled with its class. Instead of a mask, diffusion U-Nets output noise in the same size as the input noisy image, so that it can be subtracted from the input.
- When the noise is added to image *pixels*, this is called *pixel* diffusion.
- *Latent* diffusion, as used by the StableDiffusion & FLUX models, actually involves VAEs. A VAE is pre-trained on real images, then the diffusion process runs in the VAE's *latent* space, as this is less computationally expensive than running diffusion in higher-dimensional pixel space (as in *pixel* diffusion).
    - Most open-source models are *latent* diffusion models as they are relatively cheap to train and run.
- Diffusion *transformers* replace the traditional U-Net architecture of diffusion models with a transformer. The latent (i.e the compressed image) is split into patches and processed by a transformer. The main advantage over the U-Net architecture is better scalability[^iclr-blog], i.e greater performance gain as more parameters are added. The diffusion transformer architecture is used in the FLUX and SD3 models.

Before we move on to multi-modal LLMs, we should briefly discuss text conditioning. 

#### Text Conditioning

We've discussed how images can be generated, but not how we can specify the content of the image.

Separate from the image generation model, we need a text encoder; CLIP[^clip] is commonly used here. CLIP consists of an image encoder and text encoder, both trained to map images and their corresponding text description close to one another in a shared embedding space. Therefore, to condition image generation on text we take CLIP's text encoder and use it to embed a text prompt. Diffusion models have cross-attention layers in which image regions attend to text tokens so that each region draws on the words most relevant to it. But how do we get these image tokens? In a DiT the input noise latent is split into patches which are then embedded. We've mentioned that U-Nets consist of convolutional layers, but a key detail is that they also contain attention blocks which tokenise (flatten) the feature maps output by the convolutional layers.

Some models such as Stable Diffusion 3 use self-attention rather than cross-attention since they first concatenate image and text tokens into a single sequence; these are called multi-modal diffusion transformers (MMDiT). Note that text-conditioned image generation was possible before the attention mechanism was invented; text was embedded into a vector, but there was no way for a given image region to know what part of the text prompt was relevant.

Multi-modal diffusion transformers are a good link to the next section on multi-modal LLMs. Like MMDiTs, multi-modal LLMs operate on combined text-image token sequences, so in theory an MMDiT and LLM can be combined.

#### Multi-modal LLMs 
- While older image generation models consisted of a text encoder attached to a diffusion model, native multi-modal LLMs can predict the next image token in the same sequence as text.
- While the exact architecture of multi-modal-LLM-based image generation models varies a lot between companies, it's important to note that they often still involve a diffusion component. This component renders an image from the latent output by the LLM. The LLM draws on its knowledge and skills (e.g reasoning) to produce image latents that more closely match the text prompt; this is why this architecture has shown an improvement in generating images containing text.

There isn't a definitive source for which synthetic-image-generation models are the *best* at the moment. However, the top rankings of the Arena text-to-image leaderboard[^arena-leaderboard] include the following (as of 3rd August 2026):

- GPT Image 2 
- Reve 2.1 
- Google Nano Banana 2

Google's Nano Banana and GPT Image 2 are examples of multi-modal LLMs. As well as generating new images, they make editing images very easy, for example adding a generated object to a real image. The exact architecures of these examples are not published. Reve combines an LLM backbone for 'planning' and a diffusion component for 'rendering'. The Reve website[^reve] states: "*Diffusion models generate beautiful images, but they're not very intelligent or scalable. Autoregressive models (LLMs) are extremely intelligent, but...latency makes creative iteration painfully slow. Reve 2.1 leverages the best of both worlds by separating planning from rendering.*" Reve also claims to mitigate degradation caused by the accumulation of diffusion and compression artifacts which result from iterative editing. They don't explain how, but they claim "*no accumulation of artifacts whatsoever*".

For many of the entries in the leaderboard, details about the model architecture haven't been released publicly. Even if we had this information, we wouldn't be able to conclusively say whether natively multi-modal LLMs outperform other architectures due to there being too many other factors that differ in how these models are trained, e.g the amount of data and compute available to the company. It's also worth noting that the number of votes and width of confidence intervals can vary a lot on the arena leaderboard.


#### Summary: How have image-generation model architectures evolved?

<center>
 <img src="readme_images/generation_timeline.png" width='75%' />
</center>

The adversarial nature of GANs can make training unstable and lead to 'collapse'. We won't go into detail about this, but the key thing to note is that diffusion models don't suffer from the same issues. As a result, diffusion models are easier to train and can generate more diverse images. So, while GANs were state-of-the-art for a while before 2020, they were largely replaced by diffusion models.

Within the category of diffusion models, we've moved from pixel diffusion to latent diffusion to diffusion transformers (and MMDiT). As mentioned, diffusion transformers are more scalable than the original U-Net diffusion architecture.

Currently, diffusion models are arguably state-of-the-art for pixel rendering, but where a simple text encoder was once used, now the reasoning skills and knowledge of multi-modal LLMs are harnessed to produce the image latents which get rendered.

https://iclr-blogposts.github.io/2026/blog/2026/diffusion-architecture-evolution/

In a 2026 ICLR blogpost, Chen et al 

provide an interactive timeline of models and their type (e.g non-text conditioned, U-Net text-to-image or DiT text-to-image). They also created a 'model architecture explorer' which enables selection of a Hugging Face diffusion model and displays 

## How are synthetic images detected?

Purpose: understand key techniques, their pros & cons and how well we can currently detect images generated by state-of-the-art models. I want to get an overview of the existing research, how different methods perform, important considerations such as amount of data, augmentations, training processes, explainability methods. This will inform my own research questions that I will explore later on.

Mahara & Rishe provide more details on detection methods in "Methods and Trends in Detecting AI-Generated Images: A Comprehensive Review"[^detection-methods-review-1].

####  Artifact-based detection

##### Periodic upsampling artifacts

Spatial artifacts are irregularities in the image pixels, for example colour inconsistencies, texture irregularities and repeating patterns. GANs produce spatial artifacts. For example, their upsampling process can leave "periodic checkerboard patterns" (Odena et al[^odena-GAN-checkerboard] explain this well).

Corvi et al[^corvi] found similar artifacts in images produced by diffusion models. Corvi et al conducted a study around the time diffusion models were overtaking GANs. As well as investigating the "forensic traces left by diffusion models", they looked at "how current detectors, developed for GAN-generated images, perform on these new synthetic images, especially in challenging social-network scenarios involving image compression and resizing". They found that detectors trained on GAN images perform poorly on diffusion-generated images, which is perhaps due to the difference in the model-specific artifacts that detectors rely on.

While Corvi doesn't attribute a cause of diffusion-generated artifacts, their 2023 follow-up paper reported that "It is well known that many GAN-based generators leave clear traces of their processing pipeline in the images...Images generated by diffusion models appear to show artifacts of a similar nature and, arguably, a similar origin". Presumably 'similar origin' refers to the upsampling process; like GANs, U-Net diffusion models have a convolutional upsampling process, and in diffusion transformers the VAE decoder performs convolutional upsampling. They also note that "post-processing steps may significantly modify and hide the artifacts", referring to post-processing steps such as compression and resizing and highlighting that these are often applied " as soon as they are uploaded on a social network".

This study highlights 2 important ideas that we'll discuss in more detail later: the first is how new generation methods leave detectors trained on older generators redundant (lack of generalisability of detectors), and the second is the challenge of image transformations in the real-world (e.g on social media).

##### Texture inconsistencies

In 2020 Liu et al observed that "the texture of fake faces [generated by GANs] is substantially different from real ones"[^gan-texture]. They used CAM methods to explore which regions CNNs pick up on in fake images; these were found to be "texture regions, e.g skin and hair". Bias towards recognising textures (as opposed to shapes) was also observed in CNNs by Geirhos et al[^cnn-texture], although this study evaluated CNNs pre-trained on ImageNet, and fake images were not involved.

Liu et al addressed the limited receptive field of CNNs directly: alongside their analysis, they proposed Gram-Net, a CNN backbone augmented with "Gram layers" that compute global texture representations rather than relying on local convolutional features alone. 

This picks up on the idea that although locality makes CNNs well suited to identifying low-level artifacts, it also constrains what they can see (e.g global features). Vision transformers, on the other hand, better capture global features due to the use of self-attention. Some studies find that vision transformers perform better than CNNs at this task for that exact reason, and also that global features survive compression and resizing better than the local features that CNNs detect[^vit-deepfake-survey].


##### Generalisation across GANs

An important study in the area of fake-image detection was conducted in 2020 by Wang et al. In "CNN-generated images are surprisingly easy to spot... for now"[^detect-paper1], they trained a ResNet50 classifier on images from a single generator (ProGAN) and found it generalised surprisingly well to unseen CNN-based generators — not just other GANs such as StyleGAN, but also super-resolution and deepfake methods. They concluded that CNN-generated images share common artifacts, allowing a detector to transfer across architectures it was never trained on.

Two features of this study recur throughout the detection literature. The first is data augmentation: Wang et al applied Gaussian blur and JPEG compression during training, and found that this significantly improved generalisation. The importance of data augmentation is reflected in other papers in this area, and is discussed further in a later section. The second feature is the caveat in the title: "easy to spot...for now". As we'll see, the artifacts that made CNN-generated images easy to spot were not present in images generated by more modern architectures.


#### Fingerprint & reconstruction methods

The artifacts discussed in the previous section were broad categories, i.e they appeared in images generated by multiple different generators. This section will discuss artifacts produced by specific types of generator.

##### Fingerprinting

Similar to how camera sensors leave a specific noise pattern in the images they produce, some generative models leave behind a unique trace. Detecting this trace in an image enables it's source to be identified. A 2019 study by Marra et al[^gan-fingerprints] studied fingerprints produced by GANs, and found that the even same GAN architecture trained in a different way produces different fingerprints. While going beyond detection to attributing an image to a specific generator sounds promising, it still suffers from the issue of the detector needing to have seen the specific generator before. This method would therefore fail against a new state-of-the-art generator for which there is no record of a fingerprint.

##### Reconstruction-based methods

Generative models can reconstruct their own outputs than they can reconstruct real images. If a real image is passed through a pre-trained diffusion model to transform it into noise, it can't then be successfully regenerated. If the original image was diffusion-generated, then through this process it can be (approximately) reconstructed. This approach is called DIRE (diffusion reconstruction error) and was proposed by Wange et al[^DIRE] in 2023. This study found that the technique generalises across different diffusion models.

The second key reconstruction method is AEROBLADE, which was developed by Ricker et al[^AEROBLADE] in 2024. This approach specifically targets latent diffusion models, in which the diffusion process runs inside the latent space of a variational autoencoder (VAE). The key finding is that "generated images can be more accurately reconstructed by the VAE than real images, allowing for a simple detection approach based on the reconstruction error". An interesting feature of this approach is that it doesn't require training a model (DIRE requires training a binary classifier on the error map). AEROBLADE was found to be "effective" on models that were state-of-the-art when the study was conducted, e.g Stable Diffusion and Midjourney. These are more recent than the models DIRE was tested on.

Given that many state-of-the-art multi-modal-LLM-based generators have a diffusion component, this begs the question of whether DIRE or AEROBLADE work for detecting images generated by such models. There don't seem to be any studies testing this specifically.

#### Generalisation & feature-space detection

All of the detection methods we've discussed so far rely on detecting a signal that is specific to a given generator, from artifacts produced by upsampling in GANs and diffusion models to the reconstruction signature of VAEs. Such detection methods don't generalise to other generators.

In the 2023 UniversalFakeDetect paper, Ojha et al found that in the feature space of CLIP's image encoder, the vectors produced from real and fake images fall into separate groups and can therefore be classified using a nearest neighbour search. This is significant because the training data didn't distinguish real vs fake images, so there was no risk of the encoder learning generator-specific artifacts; the training data was 400 million image-text pairs, and the encoder happened to learn to encode features which differ between real and fake images.

It should be noted that the test data used in this study came from 2023-era models, so we can't say if this approach works for the models that are state-of-the-art in 2026.

The flip-side of this non-generator-specific strength is that the separation between real and fake images in the encoder's feature space is coarse and therefore not highly accurate. The logical next step was to fine-tune the CLIP encoder on real vs fake images. While this did improve accuracy[^mpft-paper], it re-introduced the issue of learning generator-specific artifacts[^dgs-net]. Two studies are notable in attempting to balance these two effects. Li et al proposed IAPL[^IAPL](image-adaptive prompt learning), which "adjusts the prompts fed into the encoder according to each testing image, rather than fixing them after training". The second study proposed GAPL[^GAPL] (generator-aware prototype learning), which fine-tunes the encoder in such a way to establish "a more robust and generalisable decision boundary".

The GAPL summarises this progression well, labelling worsening detector performance caused by increased source diversity as the "Benefit then Conflict dilemna". They attribute this to two things: "data-level homogeneity, which causes the feature distributions of real and synthetic images to increasingly overlap" and "a critical model-level bottleneck from fixed, pre-trained encoders that cannot adapt to the rising complexity".

Note that the IAPL[^GAPL] paper and GAPL[^IAPL] papers are from 2025 and 2026 respectively, and they test on images produced by GAN and diffusion-based generators.

#### Multi-modal

Accuracy aside, the feature-space detection methods discussed in the previous section simply output a binary classification (real or fake), without providing any reasoning. The need for explainability motivates the use of multi-modal LLMs. However, the accuracy of MLLMs in detecting fake images could also be impacted by hallucinations. 

Gao et al[^hallucinations] found that "evaluation results across various vision-language tasks...consistently illustrate the existence of a synthetic image-induced hallucination bias...LVLMs appear to adopt some inherent non-semantic shortcuts in synthetic images". They attribute this to models being designed for and trained on 'natural' data but don't discuss this in detail. They note a desire to understand "the cause of synthetic image-induced hallucination bias from the perspective of image synthesis mechanisms" in future work.

In a 2025 study, Ji et al[^detect-paper6] produced ~9000 AI-generated images annotated with captions explaining the visual flaws and bounding boxes showing where they're located. They fine-tuned MLLMs on this dataset, so that along with a binary label the model outputs predicted bounding boxes and a caption for each region. While high accuracy was reported, the human annotation required is labour-intensive.

In the 2025 `ThinkFake`[^detect-paper7] paper, Huang et al stated that: "methods that incorporate MLLMs for explanation typically rely on extensive supervision and lack autonomous reasoning, making it difficult to generalise to complex, unseen scenarios". In an attempt to address this, they use GRPO (group relative policy optimisation) reinforcement learning, which Huang et al say "enhances generalisation by letting the model learn how to 'think'", whereas SFT (supervised fine tuning) only "memorises". They compare the performance of an MLLM with 4 different training strategies: out-of-the-box, SFT, RL and finally SFT + RL. The accuracy scores respectively were 54.1%, 76.7%, 72.4% and 84%, i.e the combination of SFT & RL performed better than either technique alone. It's the combination of SFT + RL applied to an MLLM that the authors call `ThinkFake`. The close to 50% accuracy of the out-of-the-box MLLM was reflected in another finding of this study, 9 2024-2025-era MLLMs were tested out-of-the-box and all failed to distinguish real from fake images, scoring between 50-60% accuracy.

The authors tested ThinkFake on LOKI[^loki], a benchmark designed to evaluate MLLM performance at detecting synthetic data. The study focussed on LOKI's 'image judgement task' which requires classifying real vs fake images across categories such as person, animal and scene. Averaging over all categories, ThinkFake achieves an accuracy of 75.4%, over 10% higher than the 7 other MLLMs reported. The human score on this task is only 27.3%. The authors conclude that the high performance across categories suggests that "its reasoning and explanatory capabilities can effectively generalise to real-world challenges."

ThinkFake achieved a mean accuracy of 84% on the GenImage benchmark. This involved training on images generated by Stable Diffusion 1.4 and testing on a mixture of diffusion and GAN generators in order to test generalisation to other generator types. While the authors don't comment on this, it appears that accuracy was higher on images from the two Stable Diffusion models which are most similar to the training data. This indicates that, like the other detectors we've discussed, ThinkFake also suffers from limited generalisation to other generators.

One issue with MLLMs is their sensitivity to the wording of the prompt. Ji et al[^detect-paper8] found that changing the work `fake` to `generated` in the prompt resulted in models rejecting (i.e refusing to provide a response) less often. Combined with the potential to hallucinate, this highlights the potential unreliability of using MLLMs to classify real vs fake images. As Ji et al highlight in their 2025 paper[^detect-paper8], "*while MLLMs show promise in detecting AI-generated images, challenges remain in interpretability and alignment with human perception...ethically, ensuring transparency and accountability in detection models is critical, especially in sensitive areas like forensics and law enforcement.*"

#### Detecting images generated by the latest state-of-the-art models

A key question I have is whether any detectors have shown potential in detecting images generated by MLLM-based generators. The LOKI and GenImage datasets used in the ThinkFake study only contained images from generators as recent as 2023. LOKI only contains 2200 images produced by 13 generators, which is quite a small sample size. To truly know whether the latest image generation models can evade detection would require creating a dataset that is constantly updated each time a new generation technique is devised, and a detector that is re-trained accordingly.

A 2026 paper by Ren et al partially addresses these topics. When testing a range of detectors out-of-the-box (i.e not fine-tuned on a consistent dataset) on images generated by models released between 2020-2025, they found that the detectors performed worse on newer generators, "dropping from approximately 79% for 2020–2021 generators to around 38% for 2024 models". The generators that detectors performed worst on were "commercial APIs or recent open-source diffusion models: Flux Dev (21% mean accuracy), Firefly v4 (18%), Midjourney v7 (24%), Imagen 4 (19%), DALL-E 3 (31%)". This lag between advances in generation and detection, which results in "newly released generators consistently evading existing detectors" motivates research on "adaptive detection strategies: test-time adaptation, continual learning, and meta-learning approaches that quickly adapt to emerging threats". Continual learning refers to continuously updating detectors as new generators are developed, but this brings the challenge of retaining previous knowledge (catastrophic forgetting).

An important point that is raised in this paper's conlusion is that "technical detection represents one component alongside source authentication, digital provenance tracking, platform policies, and media literacy education". In the next section `Watermarking & C2PA` we'll briefly discuss the idea of provenance

#### Watermarking & C2PA

Some model providers such as Google add invisible watermarks to their images to enable end-users to identify the image as AI-generated (only certain models / platforms such as Google's Gemini can detect the watermark). While Google's SynthID watermark was designed to be robust to image transformations / manipulations such as compression and filtering, there is some evidence that the watermark can be removed.

C2PA is another approach to labelling AI-generated content. It uses cryptographically-signed metadata to provide secure & verifiable records of a media file's origin and changes. Currently, C2PA metadata gets stripped when downloading an image from a social media site or simply taking a screenshot of an image.

In August 2026 the EU AI Act will mandate that AI-generated image, audio and text must be tagged as AI-generated, using both a machine-readable watermark and a human-readable label.

### Timeline

Mahara & Rishe provide more details on detection methods in "Methods and Trends in Detecting AI-Generated Images: A Comprehensive Review"[^detection-methods-review-1].

There is now the question of whether it will even be possible to detect synthetic images in the future. We've seen detection methods evolve alongside the weaknesses in generation methods, from detecting low-level artifacts to semantic inconsistencies, but we're now seeing such inconsistencies disappear as generators improve.
    

# References

[^reve]: https://app.reve.com/model
[^eu-parliament]: https://www.europarl.europa.eu/RegData/etudes/BRIE/2025/775855/EPRS_BRI%282025%29775855_EN.pdf
[^arena-leaderboard]: https://arena.ai/leaderboard/text-to-image
[^iclr-blog]: https://iclr-blogposts.github.io/2026/blog/2026/diffusion-architecture-evolution/
[^clip]: https://openai.com/index/clip/
[^future-improvements]: https://arxiv.org/abs/2604.28185
[^openai-dalle]: https://openai.com/index/dall-e/
[^dalle1-1]: https://mlberkeley.substack.com/p/vq-vae?utm_source=publication-search
[^dalle1-2]: https://mlberkeley.substack.com/p/dalle2?utm_source=publication-search
[^detect-paper1]: https://arxiv.org/abs/1912.11035
[^detect-paper2]: https://www.mdpi.com/2313-433X/9/10/199
[^detect-paper3]: https://ieeexplore.ieee.org/document/10409290
[^UniversalFakeDetect-paper]: https://arxiv.org/abs/2302.10174
[^detect-paper6]: https://arxiv.org/html/2506.07045v1#S3
[^detect-paper7]: https://arxiv.org/abs/2509.19841
[^detect-paper8]: https://arxiv.org/abs/2504.14245
[^detect-paper9]: https://arxiv.org/abs/2203.11807
[^detect-paper10]: https://arxiv.org/abs/2411.19417
[^detect-paper11]: https://arxiv.org/abs/2509.21864
[^detect-paper12]: https://arxiv.org/abs/2511.21507
[^aigi-holmes-dataset]: https://huggingface.co/datasets/zzy0123/AIGI-Holmes-Dataset
[^diffusion-datasets]: https://github.com/WisconsinAIVision/UniversalFakeDetect
[^dragon-dataset]: https://huggingface.co/datasets/lesc-unifi/dragon/tree/main
[^AIS-4SD]: https://zenodo.org/records/15131117
[^SFHQ-T2I]: https://www.kaggle.com/datasets/selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models/data
[^SFHQ-part1]: https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-1
[^CocoGlide]: https://arxiv.org/abs/2212.10957
[^french-gov-paper]: https://www.peren.gouv.fr/en/perenlab/2025-02-11_ai_summit/#lenjeu-interroger-les-d%C3%A9tecteurs-%C3%A0-l%C3%A9tat-de-lart-%C3%A0-bon-escient
[^image-net]: https://www.image-net.org/update-sep-17-2019.php
[^clip-training-data]: https://voxel51.com/blog/a-history-of-clip-model-training-data-advances
[^out-of-box-detection]: https://arxiv.org/abs/2602.07814
[^community-forensics-paper]: https://arxiv.org/abs/2411.04125
[^community-forensics-dataset]: https://jespark.net/projects/2024/community_forensics/
[^detection-methods-review-1]: https://arxiv.org/abs/2502.15176
[^DIRE]: https://arxiv.org/abs/2303.09295
[^verdoliva]: https://arxiv.org/abs/2001.06564
[^gan-texture]: https://arxiv.org/abs/2002.00133
[^cnn-texture]: https://arxiv.org/abs/1811.12231
[^odena-GAN-checkerboard]: https://distill.pub/2016/deconv-checkerboard/
[^corvi]: https://arxiv.org/abs/2211.00680
[^gan-fingerprints]: https://arxiv.org/abs/1811.08180
[^AEROBLADE]: https://arxiv.org/abs/2401.17879
[^vit-deepfake-survey]: https://arxiv.org/abs/2405.08463
[^mpft-paper]: https://arxiv.org/abs/2601.03586
[^dgs-net]: https://arxiv.org/abs/2511.13108
[^IAPL]: https://arxiv.org/abs/2508.01603
[^GAPL]: https://arxiv.org/abs/2512.12982
[^hallucinations]: https://arxiv.org/abs/2403.08542
[^loki]: https://arxiv.org/abs/2410.09732
[^tasnim]: https://arxiv.org/abs/2511.02791



## Papers that may be useful

- CAN MULTI-MODAL (REASONING) LLMS WORK AS DEEPFAKE DETECTORS? (2025) https://arxiv.org/pdf/2503.20084
- A Timely Survey on Vision Transformer for Deepfake Detection (2024) https://arxiv.org/abs/2405.08463

- Community Forensics: Using Thousands of Generators to Train Fake Image Detectors (2025) [^community-forensics-paper]
    - "One of the key challenges of detecting AI-generated images is spotting images that have been created by previously unseen generative models"
    - collected 2.7 million images from 4803 different models
        - 774K images from diffusion models & GANs
        - 15K images from 'SOTA' model with unknown architectures, e.g DALL-E2, Midjourney V5, FLUX.1-dev & Imagen 3
    - study generalisation abilities of fake image detectors
    - dataset[^community-forensics-dataset]
    - "Although today’s datasets often contain millions of fake images, they come from a relatively small number of generators. As a result, this data fails to capture many sources of variation that one might encounter in the wild"
    - "In contrast to observations from recent work, we find that end-to-end training of classifiers based on CNNs or ViTs generalizes well"