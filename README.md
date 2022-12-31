# BentoML
## Deploy Sklearn Model in Production wih BentoML

### Train the model and save it in the bentoml models store
  > python 01_train.py
  

### Check model store
  > bentoml models list
  

### Serve the model Method 1: 
  > cd service/ # get in 'service.py' location first
  > 
  > bentoml serve service:iris_service --reload

### Serve the model Method 2: build a bento instance and serve it in production mode
  > bentoml build service/ 

  or
  > cd service/ # get in 'bentofile.yaml' location 
  > 
  > bentoml build 

  then 
  > bentoml serve iris_classifier:latest --production

### Request the service to  make predictions
  > python 02_request_service.py 

### Serve the model Method 3: containerize the bento with docker
  > bentoml containerize iris_classifier:tag 
  > 
  > docker images # new image built
  > 
  > docker run -p 3000:3000 iris_classifier:tag 
  
  The service is running on localhost port 3000, we can request for predictions
  > python 02_request_service.py 