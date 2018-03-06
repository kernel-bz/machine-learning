From: http://blog.postmaster.gr/2007/11/24/having-fun-with-the-mnist-database/

The MNIST database of handwritten digits, available from this page, has a
training set of 60,000 examples, and a test set of 10,000 examples. It is a
subset of a larger set available from NIST.

It consists of four files that store data in a simple file format (idx format)
which is documented in the MNIST database homepage. I wrote this C program to
be able to extract the descriptions of handwritten digits of a certain value
separately (eg. only the 0s or only the 1s). The text output is pretty simple
and closely resembles the format of a PGM file. It is describes the grayscale
pixel values of the image in decimal ASCII. Pixel values range from 0 to 255.

As a bonus, there is the opportunity to extract the images in portable graymap
file format (PGM) image files.
