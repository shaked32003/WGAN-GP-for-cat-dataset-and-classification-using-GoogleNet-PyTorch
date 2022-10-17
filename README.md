# introduction
In this project I am creating generative data using the wgan-gp model of dog and cat faces and performing binary classification for the generative data I created using the GoogleNet network
My goal in the project was to illustrate and explore the potential of using generative data for various purposes in the field of dl and dealing with the challenges when creating it, experience in the field of generative models and in particular in wgan-gp.

The following coverage presents the **main points and main conclusions** in each part of the project process, but in practice there were additional considerations

# dataset
The data area was one of the most challenging areas in the project and is divided into several parts where the data challenges changed in each of them:

**Data for wgan-gp input:**

In this part I was required to find and process data that would suit the task and it would be possible to create synthetic images of cats and dogs using the wgan model that could be worked with

I first used images of whole dogs (https://www.kaggle.com/datasets/tongpython/cat-and-dog) and had the assumption that the data I chose was indeed noisy but workable for a wgan type model
The data included about 20 thousand images in different positions of different types of dogs which I converted into 128x128 size images when they were entered into the model
The results were not good and it seemed that the model was far from succeeding in imitating the given distribution
In the bottom image you can see the original data and in the top image the performance of the model for the given data:

![Uploading צילום מסך 2022-10-06 ב-1.15.03.png…]

After researching and reading I came to the conclusion that the data is too complex and the variation between the images is large which makes the data problematic when the main reasons are:
- Many background noises that make it difficult to perform the task
- The position of the dogs varies from image to image
- A combination of the first two factors I mentioned with the fact that these are 20k images with different colors causes problems in performing the task since the variation between the images was relatively large and the number of images was large

The main conclusion was that it is better to use data that focuses more on a certain object and reach a situation where the variation between the images will not be high

I decided to focus on the faces of the dogs and cats and create images that resemble them
The new goal was to generate data consisting only of cat/dog faces
The first data I specified was not workable for the task due to the points mentioned above so I looked for new face data
Finally I chose for the dogs (https://www.kaggle.com/datasets/wutheringwang/dog-face-recognition) which included 18k images and for the cats (https://www.kaggle.com/datasets/spandan2/ cats-faces-64x64 -for-generative-models) which included 16k images when I also processed them with a resolution of 128x128 which was more appropriate in the case probably due to the continuity of colors and the low variation between the pixels


After cleaning images that I placed that could damage the quality of the output for one of the reasons I mentioned above, I got the following results where in the upper block you can see the images created by the generator and in the lower block the original images:

<img width="263" alt="צילום מסך 2022-10-14 ב-20 51 46" src="https://user-images.githubusercontent.com/96596252/196210837-f7bf39fa-c2a9-453c-aeb2-f350528359ab.png">

<img width="261" alt="צילום מסך 2022-10-17 ב-18 53 00" src="https://user-images.githubusercontent.com/96596252/196224787-46a83e43-28cf-4ace-aea2-38e549e0605f.png">

The results obtained were good relative to the running time of the model and the amount of learned parameters (I will elaborate on this in the model section) but they could not be used as input to the classification model due to their low resolution.
As a result, I decided to change direction and try to produce images with a higher resolution of 512x512.
In addition, following the findings I have reached so far, I assumed that it is possible to achieve good performance even for face images that are noisier than what I have worked with so far due to the fact that the image will be more "detailed" thanks to the increased amount of pixels and it will be possible to identify additional patterns even for high variability with a suitable model

In order to carry out the task I used the data (https://www.kaggle.com/datasets/andrewmvd/animal-faces) and in it I used the "dog" and "cat" folders that included about 5k different pictures of faces with louder ones at a resolution of 512x512
Also, the task should make changes to the network architecture, which I will mention in the "model" section

Below is an example of the results obtained:

<img width="266" alt="צילום מסך 2022-10-17 ב-16 31 16" src="https://user-images.githubusercontent.com/96596252/196223664-389617fe-bb9c-4501-8960-77c6fa520b06.png">





# Model

![IMG_4368](https://user-images.githubusercontent.com/96596252/195898370-b7fb3055-b7bd-42a8-bad9-7bcb29f7715f.PNG)

The model chosen was as mentioned wgan-gp
The architecture of the Generator was built from 5 blocks including:

- nn.ConvTranspose2d()
- nn.BatchNorm2d()
- nn.ReLU()

and another exit layer that includes:

- nn.ConvTranspose2d()
- nn.Tanh()

The architecture of the discriminator that the wgan model uses as the Critic was built from 5 blocks that include:

- nn.Conv2d()
- nn.InstanceNorm2d()
- nn.LeakyReLU()

and another exit layer that includes:

- nn.Conv2d()

After researching that the model performs well with nn.InstanceNorm2d()
and nn.LeakyReLU() in tasks similar to my task such as creating a person's face or bedrooms

since the datasets share some characteristics, I used the same hyper-parameters for both tasks

In general, the parameters I used are:

* optimize = Adam
* epochs = 80
* batch = 128
* gen_lr = 0.00035
* crit_lr = 1e-4
* z_dim = 200

where z_dim is the size of the hidden space

**In the next part I will explain my decisions for each of the parameters**


