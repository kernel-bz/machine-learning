#define GINI(pos, n) (2.*(pos)/(n)*((n)-(pos))/(n))
#define ENTROPY(pos, n) ((pos)/(n)*log((pos)/(n)) + ((n)-(pos))/(n)*log(((n)-(pos))/(n)))
#define WGINI(pos, tot) (2.*(pos)/(tot)*((tot)-(pos))/(tot))
//#define D 29
//#define N 1000000
#define D 785
#define N 13007

//Note that D = number of features + 1 (for the label)

struct TreeNode {
    struct TreeNode *parent;
    struct TreeNode *right;
    struct TreeNode *left;
    int index;
    double threshold;
    double label;
};

typedef struct TreeNode Node;

struct DataPod {
    int key;//key to identify sample points 
    double *val;//pointer to double array containing feature values for process
    double label;
    double weight;
};

typedef struct DataPod Pod;

struct HashEntry {
    struct HashEntry *next;
    int key;
    char loc;
};

typedef struct HashEntry Entry;

//data.c
double **ParHIGGS(int *feature_list, int num_features);
double **ParMNIST17(int *feature_list, int num_features);
double **MNIST17();
double **MNIST49();

//sort.c
void Sort(double **data, int first, int last, int a);
void MergeSort(double **data, int first, int last, int a);
void Merge(double **data, int first, int mid, int last, int a);
void QuickSort(double **data, int first, int last, int a);
void Partition(double **data, int first, int last, int a, int ends[]);
void PodSort(Pod **data, int first, int last, int feat);
void PodPartition(Pod **data, int first, int last, int feat, int ends[]);

//tree.c
int BestSplit(double **data, int n, int first, int col, int pos, double *impurity);
int WeightedBestSplit(double **data, int n, int first, int col, double pos, double tot, double *impurity);
int PodWBS(Pod **data, int n, int first, int feat, double pos, double tot, double *impurity);
void SplitNode(Node *node, double **data, int n, int first, int level);
void ParallelSplit(Node *node, Pod ***data, int n, int first, int level, int rank, int num_features);
void BuildTree(Node *root, double **data, int n);
void TreeFree(Node *node);
double TestPoint(Node *root, double *data);
void TreePrint(Node *node, int level);

//boost.c
double WeakLearner(Node *tree, double *data);
double Error(Node *tree, double **data, int n);
double PError(Node *tree, double **data, Pod **base, int n);
double AdaBoost(double **data, int n);





