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
* ```dreambooth``` is another example of fine tuning (?) takes an existing token and fine tunes a model to recognise that token (?) unclear on how this is different from ```embedding```

## What's Actually Going on from a Machine Learning Point of View
* stable diffusion is normally explained from a mathmatical derivation
* this will be a different way of thinking about stable diffusion
* both explanations are equally mathmatically valid
* start with example of trying to use stable diffusion to create hand-written digits
* imagine some magic api function can take any image and give a probability that an image is a handwritten digit...
* you can use this identifying api funtion to create new images of hand written digits
* you could try tweaking an individual pixel's brightness and see what the effect is on the probability
* each pixel has a gradient that represents how it could take a given pixel closer or further away from the target image (?)
* you can apply a multiplyer across all the gradients, change the image, and then re evaluate the new image
* ```finite differencing``` would require applying this process to every pixel individually
* in place of this we can use ```f.backward()``` (?) to use ```analytic derivatives``` and get ```image_x.grad``` (?) which would give us our desired output all at once (?)
* tldr → ```analytic derivatives``` method makes this more efficient but is conceptually the same process, it gets us one pass closer to an image that meets our desired criteria
* we'll write our own calculus functions later on, like f.backward
* for now we'll assume that these things exist
* ```🤔 🤔 🤔``` So we're magically getting image_x.gradient from magic_function.backward() without having to actually run our magic_function on image_x for every pixel variation. It seems like th reason we would do this repeatedly in passes rather than just maximize the gradient in one pass is because the image would change too literally on a local individual pixel level. And doing it in passes gives magic_function.backward() a chance to reassess the individual changes in the context of a more and more cohesive image. (?)
* but this all still relies on magic_function working
* right now it's just a black-box
* typically if we've got a black-box we can train a neural net to perform our desired function
* normally distributed random variable with a mean (or main ?) of zero and a variance of 0.0 to 1.0 ```N(0, 0.3)``` which in sum produces a ```mean squared error``` (?)
* so to train our neural net we feed it a set of hand-written numbers where we've added a little noise to some and a lot to others, it predicts what the noise was that we added, we take the loss between the predicted output vs the actual noise produces (the ```mean squared error```) and we use that to update the weights
* now we have a trained neural network that can identify hand written numbers and can also push static incrementally towards a new image of a hand written number
* the first component of stable diffusion is a ```unet``` which was developed for medical imaging and that we'll build ourselves later on
* the input for a ```unet``` is a somewhat noisy image and the output is the noise
* the standard input image resolution is 512 pixels x 512 pixels by 3 color channels
* we know its possible to compress images for more efficient storage (ie jpegs) so we can compress our 512x512x3 image down to smaller and smaller images with deeper and deeper dimensions and then compressing the dimensions
* these compression layers are ```neural network convolutions```
* initially when training a neural network we would perform these convolutions / compressions and then uncompress / de-convolute it and measure the loss / mean square error
* we train this model to spit out exactly what we put in
* this type of model is called an ```auto-encoder```
* the reason these are interesting is because we can put an image through just one half of this (either the encoder or the decoder half) and it will act as an effective image compression algorithm
* we call the compressed images ```latents```
* now we can compress all our input images that we want to train our ```unet``` on with ```latents``` that are much smaller than the original images
