FROM --platform=linux/amd64 rocker/r-base:latest

# Install system dependencies needed for R packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libfribidi-dev \
    libharfbuzz-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff5-dev \
    libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install BiocManager for Bioconductor packages
RUN R -e "install.packages('BiocManager', repos='https://cloud.r-project.org')"

# Install tidyverse (CRAN)
RUN R -e "install.packages('tidyverse', repos='https://cloud.r-project.org')"

# Install Bioconductor packages: ggtree and treeio
RUN R -e "BiocManager::install(c('treeio', 'ggtree'), update=FALSE, ask=FALSE)"

# Set the default command
CMD ["R"]

