# Fastai & Rust 🦀 👀 💦

Notes on the second part of fast.ai which focuses on building a stable diffusion implementation from scratch (!?). Going to try it in Rust. Doing this with no prior ML or Rust experience so we'll see. 
- [free fast.ai lectures / part-2](https://www.fast.ai/posts/part2-2022-preview.html)
- [rust docs / examples](https://doc.rust-lang.org/stable/rust-by-example/hello.html)

## Lecture 9

### Introduction
* jumping into part 2 will be tough
* knowing how to write an sdk loop (?) pytorch / tensorflow basics etc helps
* normal to spend ~10hrs working through / re-watching each lecture
* course focuses on foundations, which don't really change
* future changes will be followable
* stable diffusion requires a lot of compute power
* paying for google colab by the hour is fine
* paper gradient seems to be recommended
* [tools and notebooks](https://github.com/fastai/diffusion-nbs/blob/master/suggested_tools.md) for playing around
* [pharmapsychotic ai art tools](https://pharmapsychotic.com/tools.html)
* [lexica text to image examples](https://lexica.art/)
* you can paste in a notebook link into colab directly from a github link
* diffusers at hugging face are things related to diffusion
* paperspace and lambda labs dont delete everything

### Getting Started / Pipelines
* hugging face requires a free token
* you can save / download custom pipelines from hugging face
* diffusion models start with random noise and take steps to pull things out of that noise
* this concept will get faster but will still be fundamental
* the `guidance_scale` keyword argument controls how much attention is paid to the requested imagery vs any imagery
* if this guidance is too strong everything starts looking the same
* if its too weak it will just return any image of anything
* image-to-image pipelines start from a noisy version of the input image
* the ```strength``` parameter controls how much the output adheres to the input
* piping the output of image-to-image into another image-to-image process han have excellent results (van gogh wolf example)
* unclear on what the fine tuning pokemon example was showcasing (?)
* another example of ```fine-tuning``` is ```textual inversion``` which fine-tunes a single ```embedding```
* ```textual inversion``` trains a single token on a small number of new input images and can produce compelling results (indian watercolor example)
* 
