/**
 *  file name:  img_mnist.c
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
#include <sys/types.h>
#include <math.h>
//#include <arpa/inet.h>

#include "usr_types.h"
#include "img_mnist.h"
#include "nn.h"

i32 img_mnist_open(img_mnist_info *img)
{
    i32 ret;
    u32 idata;

    img->fd1 = open(img->fname1, O_RDONLY, 0); ///labels file
    if (img->fd1 < 0) {
        printf("error for open(%s)\n", img->fname1);
        return -1;
    }
    img->fd2 = open(img->fname2, O_RDONLY, 0); ///labels file
    if (img->fd2 < 0) {
        printf("error for open(%s)\n", img->fname2);
        return -2;
    }

    ret = read(img->fd1, &idata, sizeof(idata)); ///temp
    if (ret < 0) {
        printf("error for read\n");
        return -3;
    }
    printf("labels temp: %u\n", ntohl(idata));

    ret = read(img->fd1, &idata, sizeof(idata)); ///number
    if (ret < 0) {
        printf("error for read\n");
        return -3;
    }
    img->lnum = ntohl(idata);
    printf("labels number: %u\n", img->lnum);


    ret = read(img->fd2, &idata, sizeof(idata)); ///temp
    if (ret < 0) {
        printf("error for read\n");
        return -3;
    }
    printf("images temp: %u\n", ntohl(idata));

    ret = read(img->fd2, &idata, sizeof(idata)); ///number
    if (ret < 0) {
        printf("error for read\n");
        return -3;
    }
    img->inum = ntohl(idata);
    printf("images number: %u\n", img->inum);

    ret = read(img->fd2, &idata, sizeof(idata)); ///rows
    if (ret < 0) {
        printf("error for read\n");
        return -3;
    }
    img->irow = ntohl(idata);
    printf("images rows: %u\n", img->irow);

    ret = read(img->fd2, &idata, sizeof(idata)); ///cols
    if (ret < 0) {
        printf("error for read\n");
        return -3;
    }
    img->icol = ntohl(idata);
    printf("images cols: %u\n", img->icol);

    return img->lnum;
}

void img_mnist_close(img_mnist_info *img)
{
    close(img->fd1);
    close(img->fd2);
}

static i32 img_read(i32 fd, u8 *buf, u32 isize)
{
    i32 ret, pos, pos1, pos2;
    u32 total, sum = 0;
    static u32 cnt=0;
    total = isize;
    memset(buf, 0, total);
    do {
        pos1 = lseek(fd, 0, SEEK_CUR);
        ret = read(fd, buf, isize);
        if (ret < 0) {
            printf("read error on while loop in the img_display()\n");
            return -1;
        }
        pos2 = lseek(fd, 0, SEEK_CUR);
        pos = pos2 - pos1;
        if (pos < total) isize = (total - pos);
        sum += pos;
        buf += pos;
    } while (sum < total);

    cnt++;
    ///printf("Reading Count=%u, Size=%u Bytes\n", cnt, sum);
    return sum;
}

void img_mnist_learning(img_mnist_info *img)
{
    u8  ubyte, *buf, *p;
    u32 isize, k=0;
    i32 ret;
    f32 cost;

    isize = img->irow * img->icol;
    buf = malloc(isize);
    if (!buf) {
        printf("malloc error in the img_display()\n");
        return;
    }

    printf("images learning... please wait...\n\n");

    while (k < img->lnum) {
        ret = read(img->fd1, &ubyte, sizeof(ubyte));
        dprintf("%d: label %d:\n", k, ubyte);

        p = buf;
        ret = img_read(img->fd2, p, isize);
        if (ret < 0) {
            printf("read error on while loop in the img_mnist_learning()\n");
            break;
        }

        cost = nn_running (p, (int)ubyte, isize, LEARNING_RATE);
        if (!(k % 1000))
            printf("Learning Count:%u, Cost: %f\n", k, cost);
        k++;
    } //while

    free(buf);
}

void img_mnist_testing(img_mnist_info *img, i32 done)
{
    u8  ubyte, *buf, *p;
    c8  c=0;
    u32 isize, i, j, k=0, correct=0;
    i32 ret;

    isize = img->irow * img->icol;
    buf = malloc(isize);
    if (!buf) {
        printf("malloc error in the img_display()\n");
        return;
    }

    printf("display images:\n\n");

    while (k < img->lnum && c != 'q') {
        ret = read(img->fd1, &ubyte, sizeof(ubyte));
        printf("Testing Count:%u, Label %d:\n", k+1, ubyte);

        p = buf;
        ret = img_read(img->fd2, p, isize);
        if (ret < 0) {
            printf("read error on while loop in the img_mnist_testing()\n");
            break;
        }

        if (done) {
            correct += nn_question(buf, (int)ubyte, isize);
        } else {
            for (i=0; i < img->irow; i++) {
                for(j=0; j < img->icol; j++) {
                    if (*p == 0) printf(" ");
                    else if (*p < 128) printf("+");
                    else if (*p > 220) printf("@");
                    else printf("*");
                    p++;
                }
                printf("\n");
            }
            correct += nn_question(buf, (int)ubyte, isize);

            printf("press any key(q for quit)...");
            c = getchar();
        }
        k++;
    } //while

    printf("Accuracy: %f\n", (float)correct / (float)k);

    free(buf);
}

