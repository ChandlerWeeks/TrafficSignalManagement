# Traffic Signal Management

This Project uses cityflow and deep q-learning to find optimal conditions for traffic signal management.

## Run instructions

### Windows & Mac

In order to run the program make sure you have docker installed. Once installed open the terminal and run the following commands in the directory you of this file.

```properties
docker pull cityflowproject/cityflow:latest
```

Then you can run this to start the docker container. This will mount the current directory in a WSL instance within the docker file. CityFlow does NOT run natively on windows, so WSL is necessary, but comes within the docker container. 

```properties
docker run -it -v ${PWD}:/home/cityflow cityflowproject/cityflow:latest
```

navigate to the cityflow-project folder in WSL by running the following

```properties
cd /home/user/cityflow-project
```

Now you can run cityflow by running


```properties
python3 main.py
```
