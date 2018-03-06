/**
 *  file name:  mnist_read.c
 *  function:   mnist file read for Machine Learning
 *  author:     JungJaeJoon(rgbi3307@nate.com) on the www.kernel.bz
 */

#include <stdio.h>
#include <unistd.h>
//#include <sysexits.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
//#include <arpa/inet.h>
#include <dirent.h>
#include <sys/types.h>

#include "usr_types.h"
#include "img_mnist.h"
#include "nn.h"

static void _cnn_mnist_test(i32 done)
{
    img_mnist_info *img;
    ///i32 done = 1;   ///read weight from file

    img = malloc(sizeof(img_mnist_info));
    done = nn_init(done);
    ///done = 0;   ///learning
    if (!done) {
        printf("---------- training for learning ----------\n");
        img->fname1 = MNIST_TRAIN_LABEL;
        img->fname2 = MNIST_TRAIN_IMAGE;
        if (img_mnist_open(img) > 0)
            img_mnist_learning(img);
        img_mnist_close(img);
    }

    printf("---------- testing for answer ----------\n");
    img->fname1 = MNIST_TEST_LABEL;
    img->fname2 = MNIST_TEST_IMAGE;
    if (img_mnist_open(img) > 0) {
        img_mnist_testing(img, !done);
    }
    img_mnist_close(img);
    free(img);
    ///if (!done) nn_write("nn.wb");
    if (!done) nn_fwrite("nn.wb");
}

int main(int argc, char **argv)
{
    _cnn_mnist_test(1);

    return 0;
}
