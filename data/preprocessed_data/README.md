This folder holds some preprocessed data files.

We do the following:

```python
    pulsar_residuals, pulsar_metadata = data_loader.LoadWidebandPulsarData.read_multiple_par_tim(par_files, tim_files)

    # Also get the separation angles between all pulsars.
    ra = pulsar_metadata["RA"].to_numpy(dtype=float)
    dec = pulsar_metadata["DEC"].to_numpy(dtype=float)
    angular_separation_matrix = data_loader.LoadWidebandPulsarData.pairwise_angular_separation(ra,dec)


    # Post-process the residuals
    processed_pulsar_residuals = data_loader.LoadWidebandPulsarData.post_process_residuals(pulsar_residuals)
```


This is just a convenience, making it easier to quickly load data when testing.  