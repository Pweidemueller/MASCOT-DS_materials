FROM --platform=linux/amd64 ubuntu:20.04

# Disable interactive prompts during package installation.
ENV DEBIAN_FRONTEND=noninteractive

# Update packages and install Python 3 and pip.
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install required Python packages.
RUN pip3 install --no-cache-dir matplotlib pandas numpy biopython requests dendropy scipy

RUN ln -s /usr/bin/python3 /usr/bin/python
# Set the default command to launch Python 3.
CMD ["python3"]
