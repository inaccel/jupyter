FROM jupyter/pyspark-notebook

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
	tensorflow \
	voila

RUN jupyter labextension install \
	jupyterlab-drawio \
	jupyterlab-execute-time \
	@aquirdturtle/collapsible_headings \
	@ijmbarr/jupyterlab_spellchecker \
	@jupyterlab/fasta-extension \
	@jupyterlab/geojson-extension \
	@jupyterlab/github \
	@jupyterlab/toc \
	@jupyter-voila/jupyterlab-preview \
	@krassowski/jupyterlab_go_to_definition \
	@lckr/jupyterlab_variableinspector \
	@telamonian/theme-darcula

RUN apt-get update && \
	apt-get install -y \
		moreutils \
		jq && \
	rm -rf /var/lib/apt/lists/*

ARG JUPYTERLAB_EXTENSION_MANAGER=/opt/conda/share/jupyter/lab/schemas/@jupyterlab/extensionmanager-extension/plugin.json
ARG JUPYTERLAB_NOTEBOOK=/opt/conda/share/jupyter/lab/schemas/@jupyterlab/notebook-extension/tracker.json
ARG JUPYTERLAB_THEME=/opt/conda/share/jupyter/lab/schemas/@jupyterlab/apputils-extension/themes.json
RUN jq '.properties.enabled.default = false' ${JUPYTERLAB_EXTENSION_MANAGER} | sponge ${JUPYTERLAB_EXTENSION_MANAGER} && \
	jq '.properties.kernelShutdown.default = true | .properties.recordTiming.default = true' ${JUPYTERLAB_NOTEBOOK} | sponge ${JUPYTERLAB_NOTEBOOK} && \
	jq '.properties.theme.default = "theme-darcula" | .properties."theme-scrollbars".default = true' ${JUPYTERLAB_THEME} | sponge ${JUPYTERLAB_THEME}

ARG SPARK_DEFAULTS=/usr/local/spark/conf/spark-defaults.conf
RUN wget -qO /usr/local/spark/jars/coral-api-1.8.jar http://jfrog.inaccel.com/artifactory/libs-release/com/inaccel/coral-api/1.8/coral-api-1.8.jar && \
	wget -qO /usr/local/spark/jars/inaccel-spark-2.4.5.jar http://jfrog.inaccel.com/artifactory/libs-release/com/inaccel/inaccel-spark/2.4.5/inaccel-spark-2.4.5.jar && \
	echo 'spark.driver.extraClassPath /usr/local/spark/jars/coral-api-1.8.jar:/usr/local/spark/jars/inaccel-spark-2.4.5.jar' >> ${SPARK_DEFAULTS} && \
	echo 'spark.driver.memory         4g' >> ${SPARK_DEFAULTS} && \
	echo 'spark.master                local[4]' >> ${SPARK_DEFAULTS}

USER ${NB_UID}

COPY --chown=${NB_UID}:${NB_GID} dot .
COPY --chown=root:root slash /

CMD ["eval.sh", "start-notebook.sh", "--NotebookApp.base_url=${NB_PREFIX}", "--NotebookApp.token=''"]
