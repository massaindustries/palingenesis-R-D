FROM python:3.11.9-slim-bookworm

RUN useradd --uid 65532 --no-create-home --shell /usr/sbin/nologin executor
COPY executor.py /opt/executor.py
RUN chmod 0555 /opt/executor.py
USER 65532:65532
ENTRYPOINT ["python", "-I", "/opt/executor.py"]
