# Docker

These dockerfiles are inspirated from https://github.com/floydhub/dl-docker/

## Prerequisites
1. Install [Docker](https://docs.docker.com/engine/installation/)
2. If you plan to use the GPU version, install the [drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

## Build the image

You have to choose a name for you image 

**CPU version**
```
docker build -t [imageName] -f Dockerfile.cpu .     
```

**GPU version**
```
docker build -t [imageName] -f Dockerfile.gpu .     
```
## First run

You only need to use this command the first time you use the container. You have to use the imageName from the previous step and choose a name for your container (it can be the same as the image).

**CPU version**
```
docker run -p 8888:8888 --name [containerName] -v [absolutePathToWorkspaceFolder]:/root/workspace -it [imageName] 
```

**GPU version**
```
nvidia-docker run -p 8888:8888 --name [containerName] -v [absolutePathToWorkspaceFolder]:/root/workspace -it [imageName] 
```

## Restart the container

After running the container for the first time, you can restart it with this command

**CPU version**
```
docker start -ai [containerName]
```

**GPU version**
```
nvidia-docker start -ai [containerName]
```

## Open a bash

If you need to open a bash to your container you can use this command (the container must be running)
```
sudo docker exec -it [containerName] bash      
```