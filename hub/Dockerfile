FROM jupyterhub/k8s-hub:0.11.1

COPY --chown=root:root slash /

USER root

ARG JUPYTERHUB_PAGE=/usr/local/share/jupyterhub/templates/page.html
RUN sed -i 's/JupyterHub/InAccel Studio/g' ${JUPYTERHUB_PAGE}
