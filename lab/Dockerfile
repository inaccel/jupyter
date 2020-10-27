FROM jupyter/pyspark-notebook

COPY --chown=root:root slash /

USER root

RUN apt-get update && \
	apt-get install --no-install-recommends -y \
		curl \
		gnupg-agent \
		jq \
		moreutils \
		software-properties-common \
		zlib1g-dev && \
	curl -fsSL https://jfrog.inaccel.com/artifactory/generic/packages/gpg | apt-key add - && \
	add-apt-repository "deb [arch=amd64] https://jfrog.inaccel.com/artifactory/generic/packages/debian /" && \
	apt-get update && \
	apt-get install --no-install-recommends -y \
		coral-api \
		inaccel && \
	mkdir -m 0777 /var/lib/inaccel/shm && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/inaccel/klib.git && \
	gcc klib/ksw.c -D_KSW_MAIN -lcoral-api -lz -o /opt/conda/bin/ksw && \
	rm -rf klib

RUN git clone --depth 1 https://github.com/inaccel/lz4.git && \
	make install -C lz4/programs LN_S="ln -sf" PREFIX="/opt/conda" && \
	rm -rf lz4

RUN pip install --extra-index-url https://test.pypi.org/simple --no-cache-dir \
		coral-api \
		inaccel-h2o \
		inaccel-keras \
		inaccel-scikit-learn \
		inaccel-tf-quant-finance \
		inaccel-vitis \
		inaccel-xgboost \
		tensorflow \
		voila \
		voila-material && \
	fix-permissions ${CONDA_DIR} && \
	fix-permissions /home/${NB_USER}

ARG JUPYTER_CONFIG=${CONDA_DIR}/etc/jupyter/jupyter_config.json
ARG JUPYTERLAB_ICONS=${CONDA_DIR}/share/jupyter/lab/staging/node_modules/@jupyterlab/ui-components/style/icons
ARG JUPYTERLAB_INDEX=${CONDA_DIR}/share/jupyter/lab/static/index.html
ARG JUPYTERLAB_THEME=${CONDA_DIR}/share/jupyter/lab/themes/@telamonian/theme-darcula/index.css
ARG JUPYTERLAB_VOILA=${CONDA_DIR}/share/jupyter/nbconvert/templates/material/static
RUN jupyter labextension install \
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
		@telamonian/theme-darcula && \
	echo '{ "VoilaConfiguration": { "template": "material" } }' >> ${JUPYTER_CONFIG} && \
	cp /jupyter.svg ${JUPYTERLAB_VOILA}/voila_logo.svg && \
	cp /jupyter.svg ${JUPYTERLAB_VOILA}/voila_logo_bleu.svg && \
	cat ${JUPYTERLAB_THEME} >> ${JUPYTERLAB_VOILA}/materialize.min.css && \
	echo 'html { background-color: var(--jp-layout-color0); } button:focus { background-color: inherit; } nav .brand-logo { padding-top: 0.5rem; } .nav-wrapper { background-color: #313335 !important; } .voila-spinner-color1 { fill: #bbbbbb !important; } .voila-spinner-color2{ fill: #d97d24 !important; }' >> ${JUPYTERLAB_VOILA}/materialize.min.css && \
	cp /jupyter.svg ${JUPYTERLAB_ICONS}/jupyter && \
	echo '#jp-MainLogo svg { height: 21px; width: auto; }' >> ${JUPYTERLAB_ICONS}.css && \
	jupyter lab build && \
	jupyter lab clean && \
	sed -i 's/JupyterLab/InAccel Studio/g' ${JUPYTERLAB_INDEX} && \
	sed -i 's/rgb(90, 90, 90)/rgba(90, 90, 90, 0.3)/g' ${JUPYTERLAB_THEME} && \
	rm -rf /home/${NB_USER}/.cache/yarn && \
	rm -rf /home/${NB_USER}/.npm && \
	fix-permissions ${CONDA_DIR} && \
	fix-permissions /home/${NB_USER}

ARG JUPYTERLAB_EXTENSION_MANAGER=${CONDA_DIR}/share/jupyter/lab/schemas/@jupyterlab/extensionmanager-extension/plugin.json
ARG JUPYTERLAB_NOTEBOOK=${CONDA_DIR}/share/jupyter/lab/schemas/@jupyterlab/notebook-extension/tracker.json
ARG JUPYTERLAB_THEME=${CONDA_DIR}/share/jupyter/lab/schemas/@jupyterlab/apputils-extension/themes.json
RUN jq '.properties.enabled.default = false' ${JUPYTERLAB_EXTENSION_MANAGER} | sponge ${JUPYTERLAB_EXTENSION_MANAGER} && \
	jq '.properties.kernelShutdown.default = true | .properties.recordTiming.default = true' ${JUPYTERLAB_NOTEBOOK} | sponge ${JUPYTERLAB_NOTEBOOK} && \
	jq '.properties.theme.default = "theme-darcula" | .properties."theme-scrollbars".default = true' ${JUPYTERLAB_THEME} | sponge ${JUPYTERLAB_THEME}

ARG SPARK_DEFAULTS=/usr/local/spark/conf/spark-defaults.conf
RUN wget -qO /usr/local/spark/jars/coral-api-1.8.jar http://jfrog.inaccel.com/artifactory/libs-release/com/inaccel/coral-api/1.8/coral-api-1.8.jar && \
	wget -qO /usr/local/spark/jars/inaccel-spark-3.0.0_1.8.jar http://jfrog.inaccel.com/artifactory/libs-release/com/inaccel/inaccel-spark/3.0.0_1.8/inaccel-spark-3.0.0_1.8.jar && \
	echo 'spark.driver.extraClassPath /usr/local/spark/jars/coral-api-1.8.jar:/usr/local/spark/jars/inaccel-spark-3.0.0_1.8.jar' >> ${SPARK_DEFAULTS} && \
	echo 'spark.driver.memory         4g' >> ${SPARK_DEFAULTS} && \
	echo 'spark.master                local[4]' >> ${SPARK_DEFAULTS} && \
	rm -rf /home/${NB_USER}/.wget-hsts

USER ${NB_UID}

COPY --chown=${NB_UID}:${NB_GID} dot .

CMD ["eval.sh", "start-notebook.sh", "--NotebookApp.base_url=${NB_PREFIX}", "--NotebookApp.token=''"]
