#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "header.h"

/*
 */


void Sort(double **data, int first, int last, int col) {
    
    QuickSort(data, first, last, col);
    return;
}


void MergeSort(double **data, int first, int last, int col) {

    if (first >= last)
        return;

    int mid = (last-first)/2;
    MergeSort(data, first, first+mid, col);
    MergeSort(data, first+mid+1, last, col);
    Merge(data, first, first+mid, last, col);
    return;
}


void Merge(double **data, int first, int mid, int last, int col) {
    double **holder = malloc((last-first+1)*sizeof(double*));
    int i = first;
    int j = mid+1;
    int k = 0;
    
    while (i < mid+1 && j < last+1) {
        if (data[i][col] <= data[j][col]) {
            holder[k] = data[i];
            ++i;
        }
        else {
            holder[k] = data[j];
            ++j;
        }
        ++k;
    }
    while(i < mid+1) {
        holder[k] = data[i];
        ++i;
        ++k;
    }
    while(j < last+1) {
        holder[k] = data[j];
        ++j;
        ++k;
    }

    for (k = 0; k < last-first+1; ++k)
        data[first + k] = holder[k];

    free(holder);
    return;
}


void QuickSort(double **data, int first, int last, int col) {

/* QuickSort and Partition implements quicksort and sorts a 2-dimensional
 * array on a particular index a between indices first and last. QuickSort places
 * entries matching the pivot next to the pivot in the sorted array to minimize
 * recursive Sort calls in the case of repeated values.
 *
 * data  = array containing sample data and labels to be sorted
 * first = first unsorted row that Sort may move
 * last  = last unsorted row that Sort may move
 * col   = column/index to sort on
 *
 */

    if (first >= last)
        return;

    //Save the ends of the unsorted portions in vector.
    int ends[2];

    Partition(data, first, last, col, ends);
    QuickSort(data, first, ends[0], col);
    QuickSort(data, ends[1], last, col);
    
    return;
}


void Partition(double **data, int first, int last, int col, int ends[]) {

/* Partition implements the partition function in quicksort. It groups
 * rows that match the pivot value together to minimize recursive calls
 * to Sort. 
 *
 * data  = array containing sample data and labels to be sorted
 * first = first unsorted row that Sort may move
 * last  = last unsorted row that Sort may move
 * col   = index to sort on
 * ends  = vector with last index of lower unsorted block and first 
 *         index of higher unsorted block
 *
 */

    double pivot = data[last][col];
    int i = first;
    int j = last-1;
    int k = last;
    int l;
    double *holder;
    while (i < j) {
        if (data[i][col] < pivot)
            ++i;
        else if (data[i][col] > pivot) {
            //swap i with j, reduce j
            holder = data[j];
            data[j] = data[i];
            data[i] = holder;
            --j;
        }
        //(data[i][col] == pivot)
        else { 
            //i goes to k-1, k-1 to j, and j to i, reduce j, reduce k
            holder = data[i];
            data[i] = data[j];
            data[j] = data[k-1];
            data[k-1] = holder;
            --j;
            --k;
        }
    }

    //Now i = j.
    if (data[i][col] < pivot)
        ++i;
    else if (data[i][col] > pivot)
        --j;
    //(data[i][col] == pivot)
    else {
        holder = data[i];
        data[i] = data[k-1];
        data[k-1] = holder;
        --j;
        --k;
    }

    //Now i = j+1. j and below are less than pivot, i and above are greater.
    //Last-k+1 entries match pivot. Swap to the dividing region, starting at i.
    int matches = last-k+1;
    for (l = 0; l < matches; ++l) {
        holder = data[i+l];
        data[i+l] = data[k+l];
        data[k+l] = holder;
    }
    
    ends[0] = i-1;
    ends[1] = i + matches;

    return;
}


void PodSort(Pod **data, int first, int last, int feat) {

/* PodSort is simply QuickSort but for data stored in Pods.
 *
 * data  = pointer to array containing sample data, labels, weights in Pod form to be sorted
 * first = first unsorted row that Sort may move
 * last  = last unsorted row that Sort may move
 * feat  =  feature/index to sort on
 *
 */

    if (first >= last)
        return;

    //Save the ends of the unsorted portions in vector.
    int ends[2];

    PodPartition(data, first, last, feat, ends);
    PodSort(data, first, ends[0], feat);
    PodSort(data, ends[1], last, feat);
    
    return;
}


void PodPartition(Pod **data, int first, int last, int feat, int ends[]) {

/* PodPartition is simply Partition but for data stored in Pods.
 *
 * data  = pointer to array containing sample data, labels, weights in Pod form to be sorted
 * first = first unsorted row that Sort may move
 * last  = last unsorted row that Sort may move
 * feat  =  feature/index to sort on
 * ends  = vector with last index of lower unsorted block and first 
 *         index of higher unsorted block
 *
 */

    srand(time(NULL));
    int p_ind = rand() % (last-first+1) + first;
    Pod *holder;
    holder = data[last];
    data[last] = data[p_ind];
    data[p_ind] = holder;

    double pivot = data[last]->val[feat];
    int i = first;
    int j = last-1;
    int k = last;
    int l;
    while (i < j) {
        if (data[i]->val[feat] < pivot)
            ++i;
        else if (data[i]->val[feat] > pivot) {
            //swap i with j, reduce j
            holder = data[j];
            data[j] = data[i];
            data[i] = holder;
            --j;
        }
        //(data[i][col] == pivot)
        else { 
            //i goes to k-1, k-1 to j, and j to i, reduce j, reduce k
            holder = data[i];
            data[i] = data[j];
            data[j] = data[k-1];
            data[k-1] = holder;
            --j;
            --k;
        }
    }

    //Now i = j.
    if (data[i]->val[feat] < pivot)
        ++i;
    else if (data[i]->val[feat] > pivot)
        --j;
    //(data[i][col] == pivot)
    else {
        holder = data[i];
        data[i] = data[k-1];
        data[k-1] = holder;
        --j;
        --k;
    }

    //Now i = j+1. j and below are less than pivot, i and above are greater.
    //Last-k+1 entries match pivot. Swap to the dividing region, starting at i.
    int matches = last-k+1;
    for (l = 0; l < matches; ++l) {
        holder = data[i+l];
        data[i+l] = data[k+l];
        data[k+l] = holder;
    }
    
    ends[0] = i-1;
    ends[1] = i + matches;

    return;
}



