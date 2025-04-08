This folder holds some useful scripts for playing with and testing Argus.


* benchmark_runtime_jax :: how long does a likelihood evaluation take? How much memory does it use? For the latter, we export some profiling data to outputs/.
    * For profiling info see https://docs.jax.dev/en/latest/device_memory_profiling.html. Some useful commands are:

        `pprof -unit=mb -http=:8080 --diff_base a.prof b.prof`
        `pprof -unit=mb -text a.prof`