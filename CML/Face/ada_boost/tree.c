#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "header.h"
#include "util.h"
///#include "mpi.h"

/* Notes: Use main to test functions.
 *
 * TODO: Make threshold midpoint between last left and first right, or inf.
 *
 */


int BestSplit(double **data, int n, int first, int col, int pos, double *impurity) {

/* Returns the row/index of the table with the least impurity after splitting
 * for fixed column/feature col. Partition rows up to and including that index
 * from everything afterwards.
 *
 * data = array of data sorted on index a
 * n    = length of table (# of rows/samples)
 * col    = sorting/splitting feature/column of data
 * pos  = number of positive labels
 *
 * Thus:
 * (pos - lpos) = right pos count
 * i = total left count
 * n - i = total right count
 *
 *
 */

    int lpos = 0;

    int argmin = n - 1;//start with the whole node
    double threshold;
    double threshmin;
    double P;
    double Pmin = GINI(pos, n);//initial impurity of node

    //Tabulate impurity for each possible threshold split
    int i = 0;
    while (i < n) {

        threshold = data[first+i][col];

        while (i < n && (data[first+i][col] == threshold)) {
            if (data[first+i][D-1] > 0)
                lpos += 1;

            ++i;
        }

        //Note that points on left = i, right = n-i

        /*
        If i=n, this is the whole node and the impurity is the initial
        which is already done. i=n would cause error below.
        */
        if (i < n) {
            P = GINI(lpos, i)*i/n + GINI(pos-lpos, n-i)*(n-i)/n;

            //Save threshold/index with min impurity
            if (P < Pmin) {
                Pmin = P;
                argmin = i - 1;
                threshmin = threshold;
            }
        }
    }

    //Save the minimum impurity to compare against other indices
    *impurity = Pmin;
    return argmin;
}


int WeightedBestSplit(double **data, int n, int first, int feat, double pos, double tot, double *impurity) {

/* Returns the row/index of the table with the least impurity after splitting
 * for fixed column/feature feat. Partition rows up to and including that index
 * from everything afterwards.
 *
 * data     = array of data sorted on index a
 * n        = length of table (# of rows/samples)
 * first    = first index in the node
 * feat      = sorting/splitting feature/column of data
 * pos      = weight of positive labels
 * tot      = total weight of all labels
 * impurity = pointer to save impurity after split
 *
 * Thus:
 * (pos - lpos) = right pos count
 * i = total left count
 * n - i = total right count
 *
 *
 */

    double lpos = 0;
    double left = 0;

    int argmin = n - 1;//start with the whole node
    double threshold;
    double threshmin;
    double P;
    double Pmin = GINI(pos, tot);//initial impurity of node

    //Tabulate impurity for each possible threshold split
    int i = 0;
    while (i < n) {

        threshold = data[first+i][feat];

        while (i < n && (data[first+i][feat] == threshold)) {
            if (data[first+i][D-1] > 0)
                lpos += data[first+i][D];

            left += data[first+i][D];
            ++i;
        }

        //Note that points on left = i, right = n-i

        /*
        If i=n, this is the whole node and the impurity is the initial
        which is already done. i=n would cause error below.
        */
        if (i < n) {
            P = GINI(lpos, left)*left/tot + GINI(pos-lpos, tot-left)*(tot-left)/tot;

            //Save threshold/index with min impurity
            if (P < Pmin) {
                Pmin = P;
                argmin = i - 1;
                threshmin = threshold;
            }
        }
    }

    //Save the minimum impurity to compare against other indices
    *impurity = Pmin;
    return argmin;
}


int PodWBS(Pod **data, int n, int first, int feat, double pos, double tot, double *impurity) {

/* Pod version of WeightedBestSplit
 *
 * data     = array of data sorted by value in Pod form
 * n        = length of table (# of rows/samples)
 * first    = first index in the node
 * pos      = weight of positive labels
 * tot      = total weight of all labels
 * impurity = pointer to save impurity after split
 *
 */

    double lpos = 0;
    double left = 0;

    int argmin = first + n - 1;//start with the whole node
    double threshold;
    double threshmin;
    double P;
    double Pmin = GINI(pos, tot);//initial impurity of node

    //Tabulate impurity for each possible threshold split
    int i = first;
    while (i < first+n) {

        threshold = data[i]->val[feat];

        while ((i < first+n) && (data[i]->val[feat] == threshold)) {
            if (data[i]->label > 0)
                lpos += data[i]->weight;

            left += data[i]->weight;
            ++i;
        }

        //Note that points on left = i, right = n-i

        /*
        If i=first+n, this is the whole node and the impurity is the initial
        which is already done. i=first+n would cause error below.
        */
        if (i < first+n) {
            P = GINI(lpos, left)*left/tot + GINI(pos-lpos, tot-left)*(tot-left)/tot;

            //Save threshold/index with min impurity
            if (P < Pmin) {
                Pmin = P;
                argmin = i - 1;
                threshmin = threshold;
            }
        }
    }

    //Save the minimum impurity to compare against other indices
    *impurity = Pmin;
    return argmin;
}


void SplitNode(Node *node, double **data, int n, int first, int level) {

/* Creates two branches of the decision tree on the array data. End condition
 * creates leaf if the purity of the node is small or if there are few
 * samples on the branch of node
 *
 * node  = pointer to node in decision tree
 * data  = table of unsorted data with features and labels (with last
 *         column as the label (data[i][d-1]))
 * n     = length of table (# of rows/samples) on branch of node
 * first = first index of samples on branch of node
 * level = the depth of node in the tree
 */

    timestamp_type sort_start, sort_stop, split_start, split_stop;
    double sort_time = 0.;
    double split_time = 0.;

    int max_level = 3;
    int min_points = 6;

    node->left = NULL;
    node->right = NULL;
    node->index = -1;

    //Get initial counts for positive/negative labels
    int i;
    int pos = 0;
    double pos_w = 0;//positive weight
    double tot = 0;//total weight
    for (i = 0; i < n; ++i) {
        tot += data[first+i][D];

        if (data[first+i][D-1] > 0){
            pos += 1;
            pos_w += data[first+i][D];
        }
    }
    int neg = n - pos;
    double neg_w = tot - pos_w;

    //Declare class for node in case of pruning on child
    if (pos_w > neg_w)
        node->label = 1;
    else if (pos_w < neg_w)
        node->label = -1;
    else if (node->parent)
        node->label = node->parent->label;
    else {
        //printf("Root node is evenly balanced.\n");
        node->label = 0;
    }

    //If branch is small or almost pure, make leaf
    if (n < min_points) {
        //printf("small branch: %d points\n", n, level);
        return;
    }
    else if (level == max_level) {
        //printf("leaf node: level = max\n");
        return;
    }
    else if (pos == 0 || neg == 0) {
        //printf("pure node\n");
        return;
    }

    ///////////////TEST//////////////////
    //printf("LEVEL: %d\n", level);
    //printf("pos=%d, neg=%d, posw=%f, negw=%f, lab=%f\n", pos, neg, pos_w, neg_w, node->label);
    //printf("GINI: %f\n", GINI(pos_w, tot));
    /////////////////////////////////////


    int col;
    int row; //best row to split at for particular column/feature
    int localrow; //first + localrow = row; receives BestSplit which returns integer in [-1, n-1]
    double threshold; //best threshold to split at for column/feature
    double impurity; //impurity for best split in feature/column
    int bestcol = -1; //feature with best split
    int bestrow = first+n-1; //best row to split for best feature
    double bestthresh; //threshold split for best feature (data[bestrow][bestcol])
    double Pmin = GINI(pos_w, tot); //minimum impurity seen so far

    //Sort table. Then find best column/feature, threshold, and impurity
    for (col = 0; col < D-1; ++col) {
        //printf("\r%5d/%5d", col, D);
        //fflush(stdout);
        get_timestamp(&sort_start);
        Sort(data, first, first+n-1, col);
        get_timestamp(&sort_stop);
        get_timestamp(&split_start);
        localrow = WeightedBestSplit(data, n, first, col, pos_w, tot, &impurity);
        get_timestamp(&split_stop);
        sort_time += timestamp_diff_in_seconds(sort_start, sort_stop);
        split_time += timestamp_diff_in_seconds(split_start, split_stop);
        row = first + localrow;
        threshold = data[row][col];

        //If current column has better impurity, save col, thresh, and Pmin
        if (impurity < Pmin) {
            bestcol = col;
            bestrow = row;
            bestthresh = threshold;
            Pmin = impurity;
        }
    }
    //printf("\r           \r");
    //printf("Sort  time: %f sec\nSplit time: %f sec\n", sort_time, split_time);

    //If splitting doesn't improve purity (best split is at the end) stop
    if (bestrow == first+n-1) {
        //printf("no improvement\n");
        return;
    }


    Sort(data, first, first+n-1, bestcol);

    //For feature, threshold with best impurity, save to node attributes
    node->index = bestcol;
    node->threshold = bestthresh;

    printf("Best feature: %d, Best thresh: %f, Impurity: %f\n", node->index, node->threshold, Pmin);

    //Create right and left children
    Node *l = malloc(sizeof(Node));
    Node *r = malloc(sizeof(Node));
    l->parent = node;
    r->parent = node;
    l->right = NULL;
    l->left = NULL;
    r->right = NULL;
    r->left = NULL;

    node->left = l;
    node->right = r;

    int first_r = bestrow+1;
    int n_l = first_r - first;
    int n_r = n - n_l;

    //printf("LEFT\n");
    SplitNode(l, data, n_l, first, level+1);
    //printf("RIGHT\n");
    SplitNode(r, data, n_r, first_r, level+1);

    return;
}


void ParallelSplit(Node *node, Pod ***data, int n, int first, int level, int rank, int num_features) {

/* ParallelSplit copies SplitNode but enacts parallelized decision tree
 * construction. We use MPI and design it so that each processor holds the
 * data for one feature plus the labels and a pointer to the sample ("pod")
 * with index to be used as a key.
 *
 * Each processor has its data sorted before ParallelSplit is called.
 *
 */

    int max_level = 3;
    int min_points = 6;

    node->left = NULL;
    node->right = NULL;
    node->index = -1;

    //int tag = 21; //Timmy
    int i;
    int feat;
    int last = first+n-1;

    //Get initial counts for positive/negative labels
    int pos = 0;
    double pos_w = 0;//positive weight
    double tot = 0;//total weight

    for (i = first; i < last+1; ++i) {
        tot += data[0][i]->weight;

        if (data[0][i]->label > 0) {
            pos += 1;
            pos_w += data[0][i]->weight;
        }
    }
    int neg = n - pos;
    double neg_w = tot - pos_w;

    //Declare class for node in case of pruning on child
    if (pos_w > neg_w)
        node->label = 1;
    else if (pos_w < neg_w)
        node->label = -1;
    else if (node->parent)
        node->label = node->parent->label;
    else {
        //printf("Root node is evenly balanced.\n");
        node->label = 0;
    }

    //If branch is small or almost pure, make leaf
    if (n < min_points) {
        //printf("small branch: %d points\n", n, level);
        return;
    }
    else if (level == max_level) {
        //printf("leaf node: level = max\n");
        return;
    }
    else if (pos == 0 || neg == 0) {
        //printf("pure node\n");
        return;
    }


    int feat_row;//best row to split at for column/feature of this process
    int row = last;//best row for best feature
    int best_feat = -1;//best feature to split on, so other processes save in tree
    double threshold;//best threshold to split at for feature
    double impurity;//impurity for best split in feature
    double Pmin = GINI(pos_w, tot);//minimum impurity seen so far

    //get_timestamp(&split_start);
    for (feat = 0; feat < num_features; ++feat) {
        feat_row = PodWBS(data[feat], n, first, feat, pos_w, tot, &impurity);
        if (impurity < Pmin) {
            row = feat_row;
            threshold = data[feat][row]->val[feat];
            Pmin = impurity;
            best_feat = feat;
        }
    }

    //get_timestamp(&split_stop);
    //split_time += timestamp_diff_in_seconds(split_start, split_stop);

    //Save impurity and process rank to structure for MPI communication
    struct {
        double P; //impurity
        int R; //rank
    } in, out;

    in.P = Pmin;
    in.R = rank;

/* AllReduce to find min impurity and corresponding process (MPI_MINLOC), then
 * receive best row and hence size of next left/right nodes
 */
    MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
    MPI_Bcast(&row, 1, MPI_INT, out.R, MPI_COMM_WORLD);
    //printf("rank %d row %d\n", rank, row);
    //If splitting doesn't improve purity (best split is at the end) stop
    if (row == last) {
        //printf("no improvement\n");
        return;
    }

    MPI_Bcast(&best_feat, 1, MPI_INT, out.R, MPI_COMM_WORLD);
    MPI_Bcast(&threshold, 1, MPI_DOUBLE, out.R, MPI_COMM_WORLD);

/*
/////////////////////////////////////////////////////
    printf("CHECK:%f \n", data[42][12696]->val[42]);
    for (feat = 0; feat < num_features; feat++) {
        for (i = first; i < last; ++i) {
            if (data[feat][i]->val[feat] > data[feat][i+1]->val[feat]) {
                printf("PRESORT FAILED, i = %d, feat = %d, first = %d", i, feat, first);
                printf("%f then %f\n", data[feat][i]->val[feat], data[feat][i+1]->val[feat]);
                abort();
            }
        }
    }
    printf("Correctly Sorted.\n");
////////////////////////////////////////////////////////
*/

    int first_r = row+1;
    int n_l = row+1-first; //first_r-first
    int n_r = n - n_l;

    //For min processor, construct and broadcast list telling which node each point goes to
    char *keys = malloc(N*sizeof(char));
    for (i = 0; i < N; ++i)
        keys[i] = -1;
    if (rank == out.R) {
        for (i = first; i < row+1; ++i)
            keys[data[best_feat][i]->key] = 1;//left
        for (i = row+1; i < last+1; ++i)
            keys[data[best_feat][i]->key] = 0;//right
    }

    MPI_Bcast(keys, N, MPI_CHAR, out.R, MPI_COMM_WORLD);

    //Sort pod pointer list into ordered right node and left node
    Pod **holder = malloc(n*sizeof(Pod*));


    int l_ind;
    int r_ind;
    for (feat = 0; feat < num_features; feat++) {
        l_ind = 0;
        r_ind = 0;
        for (i = 0; i < n; ++i) {
            if (keys[data[feat][first+i]->key] == 1) {
                holder[l_ind] = data[feat][first+i];
                l_ind++;
            }
            else if (keys[data[feat][first+i]->key] == 0) {
                holder[n_l+r_ind] = data[feat][first+i];
                r_ind++;
            }
        }

        for (i = 0; i < n; ++i)
            data[feat][first+i] = holder[i];
    }


/*
/////////////////////////////////////////////////////
    for (feat = 0; feat < num_features; feat++) {
        for (i = first; i < row; ++i) {
            if (data[feat][i]->val[feat] > data[feat][i+1]->val[feat]) {
                printf("POST SORT FAILED, i = %d, feat = %d, first = %d", i, feat, first);
                printf("%f then %f\n", data[feat][i]->val[feat], data[feat][i+1]->val[feat]);
                abort();
            }
        }
        for (i = row+1; i < last; ++i) {
            if (data[feat][i]->val[feat] > data[feat][i+1]->val[feat]) {
                printf("POST SORT FAILED, i = %d, feat = %d, first = %d", i, feat, first);
                printf("%f then %f\n", data[feat][i]->val[feat], data[feat][i+1]->val[feat]);
                abort();
            }
        }
    }
    printf("Correctly POST Sorted\n");
    printf("CHECK:%f \n", data[42][12696]->val[42]);
    printf("CHECK:%f \n", data[42][12697]->val[42]);
////////////////////////////////////////////////////////
*/

    free(keys);
    free(holder);

    int p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    int remainder = (D-1)%p;
    int fpp = (D-1)/p;
    if (out.R < remainder)
        node->index = out.R*(fpp + 1) + best_feat;
    else
        node->index = remainder*(fpp + 1) + (out.R-remainder)*fpp + best_feat;

    node->threshold = threshold;

    if (rank == 0)
        printf("Best feature: %d, Best thresh: %f, Impurity %f, n_l %d\n", node->index, node->threshold, out.P, n_l);

    //Create right and left children
    Node *l = malloc(sizeof(Node));
    Node *r = malloc(sizeof(Node));
    l->parent = node;
    r->parent = node;
    l->right = NULL;
    l->left = NULL;
    r->right = NULL;
    r->left = NULL;

    node->left = l;
    node->right = r;

    //Make MPI Barrier, then begin next round; check level, entropy, or purity to decide

    //printf("LEFT\n");
    ParallelSplit(l, data, n_l, first, level+1, rank, num_features);
    //printf("RIGHT\n");
    ParallelSplit(r, data, n_r, first_r, level+1, rank, num_features);

    return;
}


void BuildTree(Node *root, double **data, int n) {

/* Wrapper function to initiate decision tree construction
 */

    SplitNode(root, data, n, 0, 0);
    return;
}


void TreeFree(Node *node) {

/* Recursively frees dynamically allocated trees
 */

    if (node == NULL)
        return;

    TreeFree(node->left);
    TreeFree(node->right);
    free(node);
    return;
}


double TestPoint(Node *root, double *data) {

    int ind;
    Node *node = root;
    while (node->left != NULL) {
        ind = node->index;
        if (data[ind] <= node->threshold)
            node = node->left;
        else
            node = node->right;
    }

    return node->label;
}


void TreePrint(Node *node, int level) {
    if (!node)
        return;

    TreePrint(node->left, level+1);
    printf("Node: level %d\n", level);
    printf("Index: %d\n", node->index);
    printf("Threshold: %f\n", node->threshold);
    printf("Class: %f", node->label);
    printf("\n");
    TreePrint(node->right, level+1);

    return;
}







