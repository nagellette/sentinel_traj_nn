### Options by file

* [raster_standardize.py](./raster_standardize.py)
    
    - **sentinel-msi:** Sentinel-MSI band standardization. Values greater than max, smaller than min are set to max/min respectively. Defined followed by min and max values in input file.
    - **zero-to-max :** Zero to max standardization except Sentinel-MSI. Values greater than max, smaller than zero are set to max/zero respectively.
    - **speed       :** Speed based trajectory standardization. Max speed used as 200km/h.
    - **min/max     :** Standardization option with min/max boundaries. Min/max values preserved.