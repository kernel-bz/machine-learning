MCC = mpicc
CC = gcc
DEP = tree.c sort.c data.c
CFLAGS = -ggdb

all: pboost uboost

pboost: paraboost.c $(DEP)
	$(MCC) $(CFLAGS) paraboost.c $(DEP) -o pboost

uboost: boost.c $(DEP)
	$(MCC) $(CFLAGS) boost.c $(DEP) -o uboost

clean:
	rm pboost uboost


