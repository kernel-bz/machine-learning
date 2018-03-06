/**
 *  file name:  img_mnist.h
 *  function:   mnist file read for Machine Learning
 *  author:     JungJaeJoon(rgbi3307@nate.com) on the www.kernel.bz
 */

#ifndef __IMG_MNIST_H
#define __IMG_MNIST_H

#include "usr_types.h"

#define MNIST_TRAIN_LABEL   "../data/train-labels-idx1-ubyte"    ///60,000개
#define MNIST_TRAIN_IMAGE   "../data/train-images-idx3-ubyte"
#define MNIST_TEST_LABEL    "../data/t10k-labels-idx1-ubyte"     ///10,000개
#define MNIST_TEST_IMAGE    "../data/t10k-images-idx3-ubyte"

typedef struct {
    c8  *fname1;    ///file name of lables for test
    c8  *fname2;    ///file name of images for test
    i32 fd1;        ///lables file for test
    i32 fd2;        ///images file for test
    u32 lnum;       ///count of lables
    u32 inum;       ///count of images
    u32 irow;       ///rows of images
    u32 icol;       ///cols of images
} img_mnist_info;

static inline u32 ntohl(u32 net_data)
{
  u8 data[4];
  memcpy(data, &net_data, 4);
  return ((((u32) data[0]) << 24) |
          (((u32) data[1]) << 16) |
          (((u32) data[2]) << 8) |
          ((u32) data[3]));
}

i32 img_mnist_open(img_mnist_info *img);
void img_mnist_close(img_mnist_info *img);
void img_mnist_learning(img_mnist_info *img);
void img_mnist_testing(img_mnist_info *img, i32 done);

#endif
