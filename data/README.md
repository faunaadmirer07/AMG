This directory is the local data root for the open-science export.

Expected subfolders used by the copied configs:

- `data/malware/sampled_malware2/`
- `data/benign/sampled_benign/`
- `data/cache/bb_data/malware2/`
- `data/malgraph_data/extracted_graphs_new/malware/sampled_malware2/`

Add your own local sample binaries and cached extractor outputs here before
running the MalConv or MalGraph pipelines.

Set `__SET_YOUR_IDA_PATH_HERE__` in the relevant config files (see the root
`README.md`) before running IDA-dependent steps.
