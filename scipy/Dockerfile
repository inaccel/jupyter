FROM jupyter/scipy-notebook

USER root

RUN apt-get update && \
	apt-get install -y \
		apt-transport-https \
		ca-certificates \
		curl \
		gnupg-agent \
		software-properties-common && \
	curl -fsSL https://jfrog.inaccel.com/artifactory/generic/packages/gpg | sudo apt-key add - && \
	sudo add-apt-repository \
		"deb [arch=amd64] https://jfrog.inaccel.com/artifactory/generic/packages/debian /" && \
	apt-get update && \
	apt-get install -y inaccel && \
	rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir coral-api inaccel-keras inaccel-scikit-learn inaccel-vitis tensorflow
RUN pip install --extra-index-url https://test.pypi.org/simple --no-cache-dir inaccel-h2o inaccel-xgboost

USER ${NB_UID}

COPY --chown=${NB_UID}:${NB_GID} dot .
COPY --chown=root:root slash /

CMD ["eval.sh", "start-notebook.sh", "--NotebookApp.base_url=${NB_PREFIX}", "--NotebookApp.token="]
