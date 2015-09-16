Reduction by key in CUDA, using warp vote and shuffle intrinsics.

The key index is a sorted list of integer indexes.

requires the following

```
git clone git@github.com:bcumming/vector.git
git clone git@github.com:bcumming/cudastreams.git
```

The results show that for single precision using warp intrinsics is only beneficial for larger bucket sizes, breaking even for buckets selected from uniform distribution 1:25.

For double precision the benefits are greater, due to the poor performance of 64 bit atomics.

## single precision

The time take to perform reduction on a single precision vector of length 2^15, or
33,554,432

The bucket size for each bucket is chosen from a uniform random distribution between 1:max_bucket

| max_bucket   |      5 |     10 |     25 |     50 |    100 |    200|
|--------------|--------|--------|--------|--------|--------|--------
| atomic       | 0.0025 | 0.0025 | 0.0036 | 0.0053 | 0.0084 | 0.0141|
| shared       | 0.0037 | 0.0040 | 0.0042 | 0.0042 | 0.0042 | 0.0042|
| shuffle      | 0.0031 | 0.0032 | 0.0033 | 0.0033 | 0.0033 | 0.0033|


## double precision

The time take to perform reduction on a single precision vector of length 2^15, or
33,554,432

The bucket size for each bucket is chosen from a uniform random distribution between 1:max_bucket

| max_bucket   |      5 |     10 |     25 |     50 |    100 |    200|
|--------------|--------|--------|--------|--------|--------|--------
| atomic       | 0.0121 | 0.0235 | 0.0582 | 0.0968 | 0.1625 | 0.3168|
| shared       | 0.0062 | 0.0064 | 0.0065 | 0.0065 | 0.0068 | 0.0075|
| shuffle      | 0.0055 | 0.0057 | 0.0056 | 0.0056 | 0.0059 | 0.0065|
