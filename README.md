# Image Segmentation using Texture and Color
1. [Dependencies](#dependencies)
2. [Algorithm](#algorithm)
3. [Files and Usage](#files-and-usage)  


### Citing `imageSegmentation`
Please cite our work if `Image Segmentation using Texture and Color` and our approach helps your research.  

```
@Misc{imageSegmentation,
  author = {{Dhall*}, A. and {Chandak*} Y.}
  title = "{Image Segmentation using Texture and Color}",
  howpublished = {\url{https://github.com/ankitdhall/imageSegmentation}},
  year = {2015}
}
```  

## Dependencies
* OpenCV for C++

## Algorithm
The proposal deals with finding objects of interest in an image. Since, an image may have many objects in different poses; classifiers on the image can’t be run directly. We will first segment the image and extract individual objects. This will help us provide a localized region which can be used as input to the classifier and eventually to identify the object of interest. Based on the low-level features defined, we can narrow down on the possible classifiers to be used.

Most of the available libraries provide functions to perform automatic segmentation based only on watershed, graph cuts and similar methods. These methods don’t take into account the texture properties of the image. For this week, we have analyzed two simple but very critical features of an image: texture and color. We have used these features to implement our own image segmentation algorithm.

First, we implemented a simple way to group similar colored regions together. It is hard to define a distance metric using the RGB color space so we converted the image to HSV color space to facilitate a simple metric to check for color similarity. Random points are selected for region growing. Pixels are clubbed together based on the color similarity metric. Once complete, we obtain a crude segmentation based on color.

Next, we find the gradient direction of individual pixels and group small patches of 20x20 px with overlaps. These patches define the texture pattern of the region by considering the frequency of each gradient direction. Statistical measures such as mean, variance, density and mode are performed on the gradient patches to quantize texture for better comparison. The regions with similar texture are merged using the same technique defined above.

Finally, we combine the above results to get a clearer segmentation of the image. This approach gives better results than the individual results from segmentation based on color or texture alone. But we were unable to define a really good similarity score that would decide when to choose texture similarity and when to consider similarity of color. We are still working on how to combine these results together. Suggestions on how we could tune or redefine the scoring metric are welcome.

The individual texture quantization and color segmentation can be used to improve other building blocks and applications.

## Files and Usage
* main.cpp - contains the initial calls and the final merge function

* color.cpp - performs segmentation based solely on color information of the image

* texture.cpp - performs segmentation based solely on texture information of the image

* histogram.cpp - genetares the histogram of 'Hue' of the image in HSV color space

* header.h - contains the function prototypes
