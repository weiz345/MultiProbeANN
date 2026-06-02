FROM ann-benchmarks

RUN pip install --no-cache-dir --upgrade pip setuptools
RUN pip install --no-cache-dir "multiprobe-ann @ git+https://github.com/weiz345/MultiProbeANN.git"
RUN python3 -c "from ann.similarity.backends.multiprobe import MultiProbeSimilarity"
