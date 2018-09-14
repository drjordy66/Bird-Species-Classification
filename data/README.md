# Data

The data can be found on the Kaggle competition site at:  
https://www.kaggle.com/c/uw-data-558-spring-20172/data

The data includes the following files: [test_images.csv](/data/test_images.csv), [train_images.csv](/data/train_images.csv), [train_labels.csv](/data/train_labels.csv), [classes.txt](/data/classes.txt), test.tgz and train.tgz. Due to storage limitations, the ".tgz" files which contain the actual images have not been included in this repository.

The following information is also available in the link above.  
## File Descriptions
- __test_images.csv__ - A list of images in the test set and the corresponding Id assigned to them
- __train_images.csv__ - A list of images in the training set and the corresponding Id assigned to them
- __train_labels.csv__ - The labels for each image (by Id) in the same format as a submission file
- __classes.txt__ - The label for each species. This duplicates information that you can find in [train_labels.csv](/data/train_labels.csv) and [train_images.csv](/data/train_images.csv), but might be convenient to use
- __test.tgz__ - The images in the test set
- __train.tgz__ - The images in the training set, arranged by species

Both the training and the test sets contain 4,320 images, with 30 from each of 144 classes. The data comes from the Caltech-UCSD Birds-200-2011 dataset here: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html.

The images you need to classify are the ones in test.tgz.
