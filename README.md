# Fastai & Rust 🦀 👀 💦

Notes on the second part of fast.ai which focuses on building a stable diffusion implementation from scratch (!?). Going to try it in Rust. Doing this with no prior ML or Rust experience so we'll see. 
- [free fast.ai lectures / part-2](https://www.fast.ai/posts/part2-2022-preview.html)
- [rust docs / examples](https://doc.rust-lang.org/stable/rust-by-example/hello.html)

<details>
  <summary>Lecture 9</summary>
  
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

### What's Going On
* What's actually going on here from a machine learning perspective?
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
* the outcoder is referred to as the ```VAE's encoder``` or ```the VAE``` which will be covered later
* but what we want is to pass in a noisy number and have it pick out and clean up an image of that number
* so now we want the neural net to take in the noise and the target number
* its going to learn to pull the number out better with the target defined as input
* we refer to whatever we're trying to remove the noise from as ```guidance``` (?)
* this approach works as is for identifying / generating numbers 1-10 but a sentence like "a cute teddy bear" is  not as straight forward
* we need a numeric representation of this sentence
* we can get word / image associations from the internet via alt text
* with millions of images / alt-texts scraped we can create two models, a ```text encoder``` and an ```image encoder```
* we can think of these for now as black-box neural nets that contain weights, which means they need inputs, outputs and a loss function
* we don't care about the specific architectures of these neural nets
* the weights for both of these models will start off as random with random output
* so both of these models output ```vectors``` or encoded representations of the respective image and text inputs
* we want these vector outputs to be similar for text that is supposed to be related to a given image
* we assess how close these vectors are by adding them together, which we refer to as the ```dot product``` of the two
* remember that these ```vectors``` are just ```3D matrices```
* conversely we want the dot product of bad image / text vectors to be low
* we can represent these dot products between image and text encoded vectors in an association table
* this table allows us to create a loss function between the sum of the diagonal dot products that represent strong image / text associations and the rest of the association table that represents bad image / text pairings (?)
* this puts encoded image and text vectors into the same space, a ```multimodal set of models```
* ```🤯 🤯 🤯```so we can feed similar sentences into our text encoder and get similar vectors when these are trained, and similar images fed into the separate image encoder will also produce similar vectors both to within the scope of the image encoder but these vectors will also be similar to those from the text encoder. That means we can encode a sentence, get a vector, decode that vector with the image encoder (decoder) and get a new image that will be associated with the original input text.
* we call this set of models a ```CLIP``` (?)
* we call the association table contrasting images and text associations ```contrastive loss``` or ```CL```
* so we refer to our text encoder as a ```clip text encoder``` which will produce similar embeddings (encoded vectors?) for similar text
* so now we have the ```unet``` that can denoise noisy latents into less noisy latents including pure noise, a ```VAE's decoder``` that can take latents and make an image and a ```CLIP text encoder``` that can take similar texts and produce similar embeddings
* sidenote: ```gradients``` are often called ```score functions```
* the language here is weird and confusing which you can hopefully ignore
* the term ```time steps``` is confusing
* ```inference``` is when you're generating an image from pure noise
* ```🤔 🤔 🤔``` When we're going from a noisier image to a less noisy image it's not the image elements that are being identified and changed, its the noise. The unet is good at identifying noise that can be eliminated, not image elements that can be amplified. (?) This is why we do it iteratively, denoising an image a bit at a time.
* a ```diffusion sampler```
* ```constant``` / ```learning rate``` (?) 
* ```momentum``` / ```adam``` => ```optimizer(s)``` (?)
* a lot of this is from the world of differential equations
* ANYWAY, the models theoretically work better if they know what percentage of the input is static, which might be unecessary / overly easy with neural nets, but this is from the world of differential equations
* ```perceptual loss``` (?)
* we're thinking of this as an ```optimization``` problem instead of a ```diferential``` problem
* next lesson we'll continue the broad overview in the notebook and then move on to implementing this with the python standard library

</details>

<details>
<summary>Lecture 10</summary>

## Lecture 10 
  
### Review & New Papers
* the 'main bit' of stable diffusion is the ```unet```
* the unet takes a noisy image and predicts the noise, then it compares to the control input, determines the loss and updates the weights
* ```distillation``` is a common practice in machine learning
* the basic idea of distillation is that you take a ```teacher network``` that knows how to do something big and complex and train a ```student network``` to do the same thing faster / better etc
* this can be as simple as taking an intermediate step in a generative diffusion (static --> image) and the final product image and training a new unet to go from the intermediate step to the end product
* this allows steps to be skipped as more and more examples of the teacher process are accumulated
* ```classifier free guided diffusion models``` cute puppy --> encoded vector --> unet --> image of a puppy // " " --> (same encoder) encoded vector --> unet --> arbitrary image // take the weighted average of the two output images, then use this for the next step of the diffusion process (?)
* this is skippable (?)
* "the ability to take any photo of a person or whatever and literally change what the person is doing is societally very important and means that anyone now can generate believable photos that never existed"
* ```TPU``` => ```tensor processing unit```

### Looking Inside the Pipeline
* remember our three critical pieces are the ```clip encoder``` the ```VAE``` and the ```unet```
* converting timesteps into the amount of noise (?) use a scheduler (?)
* first we need to tokenize the input sentence
* this translates each word into an individual number (only words its specifically trained on already?)
* ```timestep``` is just a way to look up how much noise is added to a given training image efficiently by 

### Matrix / Tensor Construction
* pick a notebook setup or work environment
* download the images of hand-written numbers dataset
* make functions to parse it into individual matrices / tensors
* find a way to open / plot these as images
* ```tensor history```
  - APL was the first language to use tensor / array math?
  - made by ```ken iverson```
  - he studied tensor analysis which is a sub-discipline of physics
  - applying an arithmetic operation to a tensor / array applies operation to each element individually
  - applying an operation between two tensors / arrays applies the operation between each respective pairing of elements
  - this all bypasses the need for iterating through matrices / arrays
* a rank 3 tensor is a 3 dimensional array
* a scalar is a 0 dimensional tensor (?)

### Random Number Function
* cloudflare uses lavalamps
* based on the ```whitman hill algorithm```
* basically we use an initial random number and track a global state that we modify

</details>

