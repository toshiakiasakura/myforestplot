FROM jupyter/datascience-notebook:python-3.9.7

WORKDIR /workdir
EXPOSE 8888

# python package installation. 
RUN pip install watermark && \
    pip install openpyxl && \
    pip install jupyterlab_vim 

RUN pip install twine && \
    pip install wheel

RUN pip install sphinx && \
    pip install sphinx_rtd_theme && \
    pip install sphinx-autodoc-typehints && \
    pip install nbsphinx && \
    pip install sphinx-sitemap

