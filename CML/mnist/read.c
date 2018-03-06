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
#include <sys/types.h>
#include <math.h>
//#include <arpa/inet.h>
///#include <winsock.h>

typedef char            c8;
typedef unsigned char   u8;
typedef unsigned int    u32;
typedef int             i32;
typedef float           f32;
typedef double          d64;

static inline u32 ntohl(u32 net_data)
{
  u8 data[4];
  memcpy(data, &net_data, 4);
  return ((((u32) data[0]) << 24) |
          (((u32) data[1]) << 16) |
          (((u32) data[2]) << 8) |
          ((u32) data[3]));
}

typedef struct _img_info {
    c8  *fname1;    ///file name of lables for test
    c8  *fname2;    ///file name of images for test
    i32 fd1;        ///lables file for test
    i32 fd2;        ///images file for test
    u32 lnum;       ///count of lables
    u32 inum;       ///count of images
    u32 irow;       ///rows of images
    u32 icol;       ///cols of images
} img_info;

static i32 img_open(img_info *img)
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
    printf("Reading Count=%u, Size=%u Bytes\n", cnt, sum);
    return sum;
}

static void img_display(img_info *img)
{
    u8  ubyte, *buf, *bp;
    c8  c=0;
    u32 isize, i, j, k=0;
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
        printf("label %d:\n", ubyte);

        bp = buf;
        ret = img_read(img->fd2, bp, isize);
        if (ret < 0) break;

        for (i=0; i < img->irow; i++) {
            for(j=0; j < img->icol; j++) {
                if (*bp == 0) printf(" ");
                else if (*bp < 128) printf("+");
                else if (*bp > 220) printf("@");
                else printf("*");
                bp++;
            }
            printf("\n");
        }

        printf("press any key(q for quit)...");
        c = getchar();
        k++;
    } //while

    free(buf);
}

static void img_close(img_info *img)
{
    close(img->fd1);
    close(img->fd2);
}

int main(int argc, char **argv)
{
    c8 *train_file_label = "../data/train-labels-idx1-ubyte";  ///label for learning
    c8 *train_file_data = "../data/train-images-idx3-ubyte";   ///data for learning
    c8 *test_file_label = "../data/t10k-labels-idx1-ubyte";    ///label for testing
    c8 *test_file_data = "../data/t10k-images-idx3-ubyte";     ///data for testing
    img_info *img;

    img = malloc(sizeof(img_info));

    if (argc > 1) {
        img->fname1 = train_file_label;
        img->fname2 = train_file_data;
    } else {
        img->fname1 = test_file_label;
        img->fname2 = test_file_data;
    }

    if (img_open(img) > 0)
        img_display(img);

    img_close(img);
    free(img);

    return 0;
}
