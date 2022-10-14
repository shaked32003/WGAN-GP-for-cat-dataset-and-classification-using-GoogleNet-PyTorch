# introduction
In this project I am creating generative data using the wgan-gp model of dog and cat faces and performing binary classification for the generative data I created using the GoogleNet network
My goal in the project was to illustrate and explore the potential of using generative data for various purposes in the field of dl and dealing with the challenges when creating it, experience in the field of generative models and in particular in wgan-gp.

The following coverage presents the **main points and main conclusions** in each part of the project process, but in practice there were additional considerations

# dataset
The data area was one of the most challenging areas in the project and is divided into several parts where the data challenges changed in each of them:

**Data for wgan-gp input:**

In this part it was required to find and process data that would suit the task and it would be possible to create synthetic images of cats and dogs through the wgan model

First I took pictures of whole dogs (https://www.kaggle.com/datasets/tongpython/cat-and-dog) and I had the assumption that the data I chose was indeed noisy but workable for a wgan type model
The data included about 20k pictures in different positions of different types of dogs
The results were not good and it seemed that the model was far from succeeding in imitating the given distribution
In the bottom image you can see the original data and in the top image the performance of the model for the given data:

<img width="262" alt="צילום מסך 2022-10-06 ב-1 15 03" src="https://user-images.githubusercontent.com/96596252/195866557-cf5b2d60-e598-4b69-b140-4faac1107cc5.png">

After researching and reading I came to the conclusion that the data is too complex and the difference between the images is large which makes the data problematic when the main reasons are:
- Many background noises that make it difficult to perform the task
- The position of the dogs changes from picture to picture
- A combination of the first two factors I mentioned with the fact that it is 20k images with different colors causes problems in performing the task since the variation between the images was relatively large and the number of images was large

The main conclusion was that you should use data that focuses more on a certain object and reach a situation where the difference between the images will not be high

I decided to focus on the faces of the dogs and cats and create images that resemble them
The new goal was to produce data consisting only of cat/dog faces
The first data I specified was not potentially processable for the task due to the above mentioned points so I looked for a new face data
Finally I chose for the dogs (https://www.kaggle.com/datasets/wutheringwang/dog-face-recognition) which included 18k photos and for the cats (https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models) which included 16k photos

After cleaning images that I assumed could damage the quality of the output for one of the reasons I mentioned above, I got the following results:


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
* **In the next part I will explain my decisions for each of the parameters**


