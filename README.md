# K-means implementations

Implementation of the k-means algorithm using serial and parallel approaches. 

## Serial

There are two files corresponding with the serial implementations:

* **Computer-serial.py**: own implenetation of the k-means algorithm employing vector operations.
* **Computer-serial-scikit.py**: scikit learn implementation of the k-means.

## Parallelization

In order to speed up the process, it is employed multiprocessing as well as threading to take advantage of the parallelization.

* **Computer-mp.py**: multiprocessing.
* **Computer-th.py**: threading.

## Data

The employed data is **computers.csv**, altough this file is very large, so it is also published **computers_dev.csv** with less samples. To generate these files the python script **computers-generator.py** can be used.
