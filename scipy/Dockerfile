FROM jupyter/scipy-notebook

USER root

RUN apt-get update && \
	apt-get install -y \
		apt-transport-https \
		ca-certificates \
		curl \
		gnupg-agent \
		software-properties-common && \
	curl -fsSL https://jfrog.inaccel.com/artifactory/generic/packages/gpg | apt-key add - && \
	add-apt-repository \
		"deb [arch=amd64] https://jfrog.inaccel.com/artifactory/generic/packages/debian /" && \
	apt-get update && \
	apt-get install -y coral-api inaccel && \
	rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/inaccel/lz4.git && \
	make install -C lz4/programs LN_S="ln -sf" PREFIX="/opt/conda" && \
	rm -rf lz4

RUN pip install --extra-index-url https://test.pypi.org/simple --no-cache-dir \
	coral-api \
	inaccel-h2o \
	inaccel-keras \
	inaccel-scikit-learn \
	inaccel-vitis \
	inaccel-xgboost \
	rise \
	tensorflow

RUN jupyter labextension install \
	jupyterlab-drawio \
	@ijmbarr/jupyterlab_spellchecker

USER ${NB_UID}

COPY --chown=${NB_UID}:${NB_GID} dot .
COPY --chown=root:root slash /

CMD ["eval.sh", "start-notebook.sh", "--NotebookApp.base_url=${NB_PREFIX}", "--NotebookApp.token=''"]
