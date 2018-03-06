/**
 *  file name:  cost_min_test.c
 *  function:   Cost Function for Machine Learning
 *  author:     JungJaeJoon(rgbi3307@nate.com) on the www.kernel.bz
 */

#include <stdio.h>
#include <stdlib.h>

typedef unsigned int    u32;    ///4Bytes
typedef int             i32;    ///4
typedef unsigned long   ul32;   ///4
typedef float           f32;    ///4
typedef double          d64;    ///8

#define ARRAY_CNT(a)    sizeof(a) / sizeof(a[0])

///hypothesis: h = w * x
#define H(w,x)  w * x

///average cost
f32 cost(f32 *xd, f32 *yd, f32 w, u32 m)
{
    u32 j;
    f32 x, y, cost = 0.0;

    for (j=0; j < m; j++) {
        x = *(xd + j);
        y = *(yd + j);
        cost += (H(w,x) - y) * (H(w,x) - y);
    }
    cost = cost / m;    ///average cost(w)
    return cost;
}

int main(void)
{
    f32 x_data[] = {1.0, 2.0, 3.0, 4.0};
    f32 y_data[] = {2.0, 4.0, 6.0, 8.0};
    f32 w, c;
    u32 icnt=0;

    for (w=-20; w < 40; w++) {
        icnt++;
        c = cost(x_data, y_data, w*0.1, ARRAY_CNT(x_data));
        printf("%4d: w=%f, cost=%f\n", (i32)w, w*0.1, c);
    }
}
