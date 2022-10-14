
# dataset
The data area was one of the most challenging areas in the project and is divided into several parts where the data challenges changed in each of them:

**Data for wgan-gp input:**

In this part it was required to find and process data that would suit the task and it would be possible to create synthetic images of cats and dogs through the wgan model

First I took pictures of whole dogs (__) and I had the assumption that the data I chose was indeed noisy but workable for a wgan type model
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
Finally I chose for the dogs ___ which included 18k photos and for the cats ___ which included 16k photos

After cleaning images that I assumed could damage the quality of the output for one of the reasons I mentioned above, I got the following results:


The following overview presents the main points and main conclusions in each part of the project process
