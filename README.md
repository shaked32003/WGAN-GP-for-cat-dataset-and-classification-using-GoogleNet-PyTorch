# WGAN-GP for cat dataset and classification using GoogleNet

# introduction
A self-study project in which I am creating generative data using the wgan-gp model of cat faces and performing binary classification for the generative data and dog image I created using the GoogleNet network using pytorch library.
My goal in the project was to illustrate and explore the potential of using generative data for various purposes in the field of dl and dealing with the challenges when creating it, experience in the field of generative models and in particular in wgan-gp.

<img width="262" alt="צילום מסך 2022-10-18 ב-12 53 23" src="https://user-images.githubusercontent.com/96596252/196410147-ec64a470-c7f4-49a9-b04e-ead990f92678.png"> 
 


The following coverage presents the **main points and main conclusions** in each part of the project process, but in practice there were additional considerations

# dataset
The data area was one of the most challenging areas in the project and is divided into several parts where the data challenges changed in each of them:

**Data for wgan-gp input:**

In this part I was required to find and process data that would suit the task and it would be possible to create synthetic images of cats using the wgan model that could be worked with

I first used images of (https://www.kaggle.com/datasets/tongpython/cat-and-dog) and had the assumption that the data I chose was indeed noisy but workable for a wgan type model
The data included about 20 thousand images in different positions of different types of cat which I converted into 128x128 size images when they were entered into the model
The results were not good and it seemed that the model was far from succeeding in imitating the given distribution


After researching and reading I came to the conclusion that the data is too complex and the variation between the images is large which makes the data problematic when the main reasons are:
- Many background noises that make it difficult to perform the task
- The position of the cats varies from image to image
- A combination of the first two factors I mentioned with the fact that these are 20k images with different colors causes problems in performing the task since the variation between the images was relatively large and the number of images was large

The main conclusion was that it is better to use data that focuses more on a certain object and reach a situation where the variation between the images will not be high

I decided to focus on the faces of cats and create images that resemble them
The new goal was to generate data consisting only of cat faces
The first data I specified was not workable for the task due to the points mentioned above so I looked for new face data
Finally I chose for the cats (https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models) which included 16k images when I also processed them with a resolution of 128x128 which was more appropriate in the case probably due to the continuity of colors and the low variation between the pixels


After cleaning images that I placed that could damage the quality of the output for one of the reasons I mentioned above, I got the following results where in the upper block you can see the images created by the generator and in the lower block the original images:

128x128 outpot:

<img width="263" alt="צילום מסך 2022-10-14 ב-20 51 46" src="https://user-images.githubusercontent.com/96596252/196210837-f7bf39fa-c2a9-453c-aeb2-f350528359ab.png">  
 

The results obtained were good relative to the running time of the model and the amount of learned parameters (I will elaborate on this in the model section) but they could not be used as input to the classification model due to their low resolution.
As a result, I decided to change direction and try to produce images with a higher resolution of 512x512.
In addition, following the findings I have reached so far, I assumed that it is possible to achieve good performance even for face images that are noisier than what I have worked with so far due to the fact that the image will be more "detailed" thanks to the increased amount of pixels and it will be possible to identify additional patterns even for high variability with a suitable model

In order to carry out the task I used the data (https://www.kaggle.com/datasets/andrewmvd/animal-faces) and in it I used the "cat" folders that included about 5k different pictures of faces with louder ones at a resolution of 512x512
Also, the task should make changes to the network architecture, which I will mention in the "model" section

Below is an example of the results obtained:

512x512 cat outpot:

<img width="266" alt="צילום מסך 2022-10-17 ב-16 31 16" src="https://user-images.githubusercontent.com/96596252/196223664-389617fe-bb9c-4501-8960-77c6fa520b06.png"> 

**data for googleNet input:**

I began to produce a relatively large number of images and save them as `jpg files`. After creation, I went through the collection of images and cleaned images from the data that could mislead the model - at this stage I looked for images whose outlines would be clear so that the model would have good working material during identification

After creating generative images of cats and clean data, I built two different data sources for the training phase:
The first was data that included 2500 images of dogs and 2500 images of cats, **with 85% of the cat data being generative images of cats** (2125 generative images)

The second was 2500 images of dogs and cats where here both classes were real and taken from the training data of (https://www.kaggle.com/datasets/andrewmvd/animal-faces)

AAt the moment my two data sets were balanced in terms of their size. It is worth checking how the model behaves with the distribution of the generative data. The augmentation methods you used were methods like `transforms.RandomHorizontalFlip()` which did not change the distribution of the data

I applied the same transforms to both training groups so that the only variable between the two classes was only that one of them had a generative dataset

# Model

Here, too, it was convenient to break down the explanations of the model into two main parts

**WGAN model:**

![IMG_4368](https://user-images.githubusercontent.com/96596252/195898370-b7fb3055-b7bd-42a8-bad9-7bcb29f7715f.PNG)

The model chosen was as mentioned wgan-gp
for 128x128 input The architecture of the Generator was built from 5 blocks including:

- `nn.ConvTranspose2d()`
- `nn.BatchNorm2d()`
- `nn.ReLU()`

and another exit layer that includes:

- `nn.ConvTranspose2d()`
- `nn.Tanh()`

The architecture of the discriminator that the wgan model uses as the Critic was built from 5 blocks that include:

- `nn.Conv2d()`
- `nn.InstanceNorm2d()`
- `nn.LeakyReLU()`

and another exit layer that includes:

- `nn.Conv2d()`

After researching that the model performs well with `nn.InstanceNorm2d()`
and `nn.LeakyReLU()` in tasks similar to my task such as creating a person's face or bedrooms

for 512x512 input I added two additional block layers to the generator and critic with the same block architecture as the one specified above (a total of 7 blocks instead of 5 and output layers the same as specified above).
increasing the resolution of the input caused an increase in the complexity of the model and, of course, a larger input dimension (x3) which increased the amount of studied parameters significantly and was felt in the running times, but nevertheless I decided to make the change.

since the datasets share some characteristics, I used the same hyper-parameters for both tasks

In general, the parameters I used are:

* optimize = Adam
* epochs = 130
* batch = 64
* gen_lr = 0.00008
* crit_lr = 1e-4
* z_dim = 200

where `z_dim` is the size of the hidden space

**In the next part I will explain my decisions for each of the parameters**

**GoogleNet model**

![Inceptionv1_architecture](https://user-images.githubusercontent.com/96596252/196228960-87a6aa84-cd17-42f8-bff5-b069d44889ed.png)

My assumption for the generative images was that they are not sharp "and the limits" that the model must learn (for example the shape of the ear) were less accurate than a real image, so I understood that a slightly deeper network than the standard mode is needed, but it is worth trying to save as much on the number of studied parameters and the training time as possible, I decided to use in the googlenet model

In addition, due to the constraints of the low amount of data that I used, the model that I claimed was pre-training in order to get an initialization that approximates the layers of the model

The parameters I chose for the model are:

* optimize = Adam
* epochs = 10
* batch = 32
* lr = 1e-3

# Training

**wgan-gp training:**

As you can understand from what I have written so far, the attempts were many and with them also the training, when the main challenge for each type of data was to balance the model for convergence
Already in the first training I saw that the lr value greatly affects the performance and the convergence and the changes in the result are significant, every smallest change in the lr values went through when at the end I made use of the parameters I mentioned in the model phase
Another criterion that had a great impact was the instability of the Adan optimizer in the first gradient steps due to a lack of calculation data when this, in the limit of the delicate lr problem, caused very large steps in the initial 100 steps - one of the solutions I tried was to switch to the RMSprom optimizer, which indeed solved the initial instability problem but had difficulty converging later in the training.

Another attempt to stabilize the convergence was to reset the weights of the model to a normal distribution, but in the end I decided not to use this because they did not contribute.

Finally I decided to stay with Adan when combining it with finding the optimal small lr value was able to produce relatively reasonable steps in the stages of Adan's instability

In the following pictures you can see the learning process of the cat pictures during the training:

epoch 10

<img width="266" alt="צילום מסך 2022-10-17 ב-19 42 25" src="https://user-images.githubusercontent.com/96596252/196235340-7ba4f98e-65fb-4891-a403-f672f7ef1b3f.png"> 

 
epoch 40

<img width="262" alt="צילום מסך 2022-10-17 ב-19 42 53" src="https://user-images.githubusercontent.com/96596252/196235399-f2508d2f-3a19-4ca6-bc22-eaf06dc78c30.png"> 
 
epoch 100

<img width="254" alt="צילום מסך 2022-10-17 ב-19 42 13" src="https://user-images.githubusercontent.com/96596252/196235308-734f3203-bc5c-41a2-80ee-eeeecc0953ae.png">

**GoogleNet training:**

In the training phase I ran two identical models for each dataset I created in order to check the behavior of the model between the generative data and the real one
After a bit of playing with the hyperparameters I was able to get almost identical to completely identical results between the two datasets of 99.9% accuracy of the generative data work
It seemed from all the attempts that in the very initial training stages of the generative data the model tends to get more confused in the val stage than in the training of the real data but the model stabilized to high accuracy percentages quite quickly
It seemed from all the attempts that in the very initial training stages of the generative data the model tends to get more confused in the val stage than in the training of the real data but the model stabilized to high accuracy percentages quite quickly

<img width="600" alt="צילום מסך 2022-11-28 ב-12 20 34" src="https://user-images.githubusercontent.com/96596252/204253683-ab499281-009c-40de-a402-384a99b2c1e6.png">

<img width="603" alt="צילום מסך 2022-11-28 ב-12 20 48" src="https://user-images.githubusercontent.com/96596252/204253742-2132dec2-d07c-46fd-829b-c5f4250514f6.png">

# conclusions
Integrating generic data into my original data can be a solution for augmentation or class balance
I believe that in the future I will continue to perform more experiments in this project. It is worth examining additional effects of using generic data in classification problems
