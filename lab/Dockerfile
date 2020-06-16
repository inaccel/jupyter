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
	tensorflow

RUN jupyter labextension install \
	jupyterlab-drawio \
	jupyterlab-execute-time \
	jupyterlab-theme-toggle \
	jupyterlab-topbar-extension \
	jupyterlab-topbar-text \
	@ijmbarr/jupyterlab_spellchecker \
	@jupyterlab/fasta-extension \
	@lckr/jupyterlab_variableinspector

RUN apt-get update && \
	apt-get install -y \
		moreutils \
		jq && \
	rm -rf /var/lib/apt/lists/*

ARG JUPYTERLAB_EXECUTE_TIME=/opt/conda/share/jupyter/lab/schemas/@jupyterlab/notebook-extension/tracker.json
ARG JUPYTERLAB_TOPBAR_TEXT=/opt/conda/share/jupyter/lab/schemas/jupyterlab-topbar-text/plugin.json
RUN jq '.properties.recordTiming.default = true' ${JUPYTERLAB_EXECUTE_TIME} | sponge ${JUPYTERLAB_EXECUTE_TIME} && \
	jq '.properties.editable.default = false | .properties.text.default = "InAccel Cloud"' ${JUPYTERLAB_TOPBAR_TEXT} | sponge ${JUPYTERLAB_TOPBAR_TEXT}

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
