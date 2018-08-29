# OWLPY

An open source time series library for Python implementing the [Matrix Profile](http://www.cs.ucr.edu/~eamonn/MatrixProfile.html).

The Matrix Profile is a time-series analysis tool that can be used for several tasks e.g. motif/discord discovery.

Original project started by [Juan](https://github.com/jbeleno).

## Installation

Clone this repo and run:
```python
python setup.py install 
```

## Test

To test the library, in the source directory run:
```python
python test_motif_discovery.py 
```
to discover motifs and discords in the ECGFiveDays dataset, or

```python
python test_query_matching.py 
```

to match a query subsequence on the Coffee datset.

Both dataset are from the [UCR Archive](http://timeseriesclassification.com/dataset.php).

## Usage

To use OWLPY, simply import from the _core_ package, as in:
```python
from owlpy.core import *
import numpy as np 
ts = np.random.rand(100)
query = np.random.rand(50)
MP, I = stamp(ts,query,15)
```
