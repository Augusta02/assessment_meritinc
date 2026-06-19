## Partial Cross Entropy Loss for Weakly Supervised Remote Sensing Segmentation

## Introduction

**Semantic Segementation** is task used to assign class lable to every single pixel in an image, in remote sensing, this means classifying each pixel os a satellye image as one 
of several land types (water, agriculture, road, ect). It is an intensive annotation task because training models are manuallu outlined for every region in the image. 

The standard approach for this task reauires dense masks, where every pixel in the image carries a label, therefore a 1024 X 1024 satellite image has over 1 million labels per image, collecting this amount
of annotation at scale is expensive and time consuming, also prone to errors, lastly very impractical to organisation working with large geographhic datasets. 

Using Sparse Point Labels to train Semantic Segementation
Our script explores an alternative method of completing tis annotation task, train our model using sparse point labels, where a humman annotator provides a small number of points from the pixels. 
How much can our model learn to segment an entire image from a handful of labelled pixels,
How much performance is lost compared to full dense supervision


## Dataset

Using the **LoveDA** dataset, publicly available remote sensing benchmark containing aerial images of both urban and rural scenes in China
Each image is 1024x1024 pixels and comes with a full pixel-level annotation mask covering seven land cover classes. 

| Class Index | Class Name
| ------ | ------
| 0 |  Background 
| 1 | Building
| 2 |  Road
| 3 |  Water
| 4 |  Barren
| 5 | Forest
| 0 |  Agriculture

The dataset contains 4,191 image-mask pairs in total. We split these into training (70%), validation (15%), and test (15%) sets using a 42 random seed for reproducibility.
One important data preprocessing step was remapping the original mask values. LoveDA stores class indices starting from 1 (values 1 to 7), but PyTorch's cross-entropy loss expects indices starting from 0. We applied a simple shift during loading:

``` mask = np.clip(mask - 1, 0, 6).astype(np.uint8) ```

## Method

### Simulating Point Labels

The LoveDA dataset provides full dense masks, but our goal is to simulate a weakly-supervised scenario. Created a helper function called **simulate_point_labels** that takes a full mask and randomly samples N pixels per class, returning a binary point mask of the same spatial size. The point mask contains 1 at labeled pixel locations and 0 everywhere else.

```
def simulate_point_labels(mask, num_points_per_class=5):
    point_mask = np.zeros_like(mask)
    classes = np.unique(mask)
    for cls in classes:
        rows, cols = np.where(mask == cls)
        n_sample = min(num_points_per_class, len(rows))
        chosen = np.random.choice(len(rows), size=n_sample, replace=False)
        point_mask[rows[chosen], cols[chosen]] = 1
    return point_mask
```
## Partial Cross-Entropy Loss

The standard cross-entropy loss computes a prediction error at every pixel and averages over the entire image. This works well when we have full dense masks but becomes problematic with sparse labels, because pixels with no annotation would contribute meaningless gradients to the training process.

Partial Cross-Entropy solves this by masking out unlabeled pixels before computing the loss. The formula is:






































