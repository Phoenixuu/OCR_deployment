FROM ubuntu:20.04

# Install dependencies
Run apt-get update && apt-get install -y\
	cmake \
	g++ \ 
	wget \
	git \
	liopencv-dev

# Copy the project
WORKDIR /app
COPY . .

# Build the project
RUN mkdir build && cd build && cmake .. && make

CMD ["./build/OCRApp"]