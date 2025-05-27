FROM deepnote/python:3.11-datascience

RUN apt-get update -qq \
    && apt-get install -y graphviz libgraphviz-dev \
    && apt-get clean

RUN pip install \
    --no-cache-dir \
    -c https://tk.deepnote.com/constraints3.11.txt \
    cvxpy highspy matplotlib networkx numpy pygraphviz
