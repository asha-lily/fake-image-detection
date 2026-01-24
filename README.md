# Fake Image Detection: Research Notes

### Motivation

The motivation for this research project comes from the increasing prevalence of AI-generated images in various aspects of everyday life and my concerns about the potential impacts of society being able to easily generate and share such content. To name just a few of these concerns:

- Online scams where the buyer relies on an image
- Fabricated political / high-profile events (and other types of misinformation)
- Non-consensual intimate imagery

### Aims

The technology used to generate images, video, audio etc is advancing faster than our ability to reliably detect synthetic content. I'd like to learn more about both how this content - specifically images - is generated and how we can detect it. 

Once I have an understanding of the current state of research in this area, I plan to run experiments of my own, fine-tuning open-source models to classify real vs synthetic images. Given time and computational resource constraints, my aim won't be to produce the best model possible, but rather to see what can be achieved with models such as CNNs and transformers.

I'm also interested in using class activation map methods to visualise image artifacts that models learn in order to distinguish real from synthetic. 