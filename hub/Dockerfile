FROM jupyterhub/k8s-hub:1.0.0

COPY --chown=root:root slash /

USER root

ARG JUPYTERHUB_PAGE=/usr/local/share/jupyterhub/templates/page.html
RUN sed -i 's/JupyterHub/InAccel Studio/g' ${JUPYTERHUB_PAGE}
