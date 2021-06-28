FROM inaccel/buildpack-deps:grpc AS coral-api

RUN git clone --depth=1 --single-branch https://github.com/inaccel/coral-api /tmp/coral-api \
 && make --directory=/tmp/coral-api src/main/resources/native/libcoral-api.so

FROM maven:3-jdk-8 AS spark

RUN git clone --depth=1 --single-branch https://github.com/inaccel/coral-api /tmp/coral-api

COPY --from=coral-api /tmp/coral-api/src/main/resources/native/libcoral-api.so /tmp/coral-api/src/main/resources/native/libcoral-api.so

RUN mvn -f /tmp/coral-api install

RUN git clone --depth 1 https://github.com/inaccel/spark.git /tmp/spark && \
	mvn -f /tmp/spark package

FROM jupyter/pyspark-notebook:aec555e49be6

COPY --chown=root:root slash /

USER root

RUN apt-get update && \
	apt-get install --no-install-recommends -y \
		curl \
		gnupg-agent \
		jq \
		libtinfo-dev \
		moreutils \
		software-properties-common \
		zlib1g-dev && \
	curl -fsSL https://setup.inaccel.com/gpg | apt-key add - && \
	add-apt-repository "deb [arch=amd64] https://dl.cloudsmith.io/public/inaccel/stable/deb/ubuntu $(lsb_release -cs) main" && \
	apt-get update && \
	apt-get install --no-install-recommends -y \
		coral-api \
		inaccel-cli \
		inaccel-openssl && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/inaccel/klib.git && \
	gcc klib/ksw.c -D_KSW_MAIN -lcoral-api -lz -o /opt/conda/bin/ksw && \
	rm -rf klib

RUN git clone --depth 1 https://github.com/inaccel/lz4.git && \
	make install -C lz4/programs LN_S="ln -sf" PREFIX="/opt/conda" && \
	rm -rf lz4

RUN pip install --extra-index-url https://test.pypi.org/simple --no-cache-dir \
		ipython-sql \
		nbresuse \
		psycopg2-binary \
		tensorflow==2.4.* \
		voila \
		voila-material && \
	fix-permissions ${CONDA_DIR} && \
	fix-permissions /home/${NB_USER}

RUN git clone --branch configurable_allocator --depth 1 https://github.com/mattip/numpy.git && \
	git -C numpy tag v1.22.0 && \
	pip install --no-cache-dir ./numpy && \
	rm -rf numpy

RUN git clone --depth 1 https://github.com/inaccel/numpy-allocator.git && \
	pip install --no-cache-dir ./numpy-allocator && \
	rm -rf numpy-allocator

RUN git clone --depth 1 https://github.com/inaccel/coral-api.git && \
	mkdir -p coral-api/src/main/python/inaccel/coral/native && \
	cp /usr/lib/libcoral-api.so coral-api/src/main/python/inaccel/coral/native/libcoral-api.so && \
	pip install --no-cache-dir ./coral-api && \
	rm -rf coral-api

RUN git clone --depth 1 https://github.com/inaccel/keras.git && \
	pip install --no-cache-dir ./keras && \
	rm -rf keras

RUN git clone --depth 1 https://github.com/inaccel/scikit-learn.git && \
	pip install --no-cache-dir ./scikit-learn && \
	rm -rf scikit-learn

RUN git clone --depth 1 https://github.com/inaccel/tf-quant-finance.git && \
	pip install --no-cache-dir ./tf-quant-finance && \
	rm -rf tf-quant-finance

RUN git clone --depth 1 https://github.com/inaccel/Vitis_Libraries.git && \
	pip install --no-cache-dir ./Vitis_Libraries/quantitative_finance/L3/python && \
	rm -rf Vitis_Libraries

RUN git clone --branch coral --depth 1 --recursive https://github.com/inaccel/xgboost.git && \
	make -C xgboost patch && \
	make -C xgboost/xgboost all && \
	cd xgboost/xgboost/python-package && \
	python setup.py install && \
	cd ../../.. && \
	rm -rf xgboost

ARG JUPYTER_CONFIG=${CONDA_DIR}/etc/jupyter/jupyter_config.json
ARG JUPYTERLAB_ICONS=${CONDA_DIR}/share/jupyter/lab/staging/node_modules/@jupyterlab/ui-components/style/icons
ARG JUPYTERLAB_INDEX=${CONDA_DIR}/share/jupyter/lab/static/index.html
ARG JUPYTERLAB_THEME=${CONDA_DIR}/share/jupyter/lab/themes/@telamonian/theme-darcula/index.css
ARG JUPYTERLAB_VOILA=${CONDA_DIR}/share/jupyter/nbconvert/templates/material/static
RUN jupyter labextension install \
		jupyterlab-execute-time \
		jupyterlab-system-monitor \
		jupyterlab-topbar-extension \
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
	echo '{ "ResourceUseDisplay": { "disable_legacy_endpoint": false, "track_cpu_percent": true }, "VoilaConfiguration": { "template": "material" } }' >> ${JUPYTER_CONFIG} && \
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
ARG JUPYTERLAB_SYSTEM_MONITOR=${CONDA_DIR}/share/jupyter/lab/schemas/jupyterlab-system-monitor/plugin.json
ARG JUPYTERLAB_THEME=${CONDA_DIR}/share/jupyter/lab/schemas/@jupyterlab/apputils-extension/themes.json
RUN jq '.properties.enabled.default = false' ${JUPYTERLAB_EXTENSION_MANAGER} | sponge ${JUPYTERLAB_EXTENSION_MANAGER} && \
	jq '.properties.kernelShutdown.default = true | .properties.recordTiming.default = true' ${JUPYTERLAB_NOTEBOOK} | sponge ${JUPYTERLAB_NOTEBOOK} && \
	jq '.properties.memory.default.label = "RAM: "' ${JUPYTERLAB_SYSTEM_MONITOR} | sponge ${JUPYTERLAB_SYSTEM_MONITOR} && \
	jq '.properties.theme.default = "Darcula" | .properties."theme-scrollbars".default = true' ${JUPYTERLAB_THEME} | sponge ${JUPYTERLAB_THEME}

COPY --from=spark /tmp/spark/jars/* /usr/local/spark/jars/

ARG SPARK_DEFAULTS=/usr/local/spark/conf/spark-defaults.conf
RUN echo 'spark.driver.extraClassPath /usr/local/spark/jars/coral-api-2.0.0.jar:/usr/local/spark/jars/jna-5.8.0.jar:/usr/local/spark/jars/netty-buffer-4.1.63.Final.jar:/usr/local/spark/jars/inaccel-spark-3.0.2_2.0.0.jar:/usr/local/spark/jars/netty-all-4.1.47.Final.jar:/usr/local/spark/jars/netty-common-4.1.63.Final.jar' >> ${SPARK_DEFAULTS} && \
	echo 'spark.driver.memory         4g' >> ${SPARK_DEFAULTS} && \
	echo 'spark.master                local[4]' >> ${SPARK_DEFAULTS}

ENV C_INCLUDE_PATH=/opt/conda/include
ENV CPLUS_INCLUDE_PATH=/opt/conda/include
ENV LIBRARY_PATH=/opt/conda/lib

USER ${NB_UID}

COPY --chown=${NB_UID}:${NB_GID} dot .

CMD ["eval.sh", "start-notebook.sh", "--NotebookApp.base_url=${NB_PREFIX}", "--NotebookApp.token=''"]
