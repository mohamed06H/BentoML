# BentoML
## Deploy Sklearn Model in Production with BentoML

### Train the model and save it in the bentoml models store
```bash
python 01_train.py
```  
```python
# Save an sklearn model in bentoml:
bentoml.sklearn.save_model(model_name, model)
``` 

### Check model store
```bash
bentoml models list
```  

### Serve the model Method 1: 
```bash
cd service/ # get in 'service.py' location first
   
bentoml serve service:iris_service --reload
```
### Serve the model Method 2: build a bento instance and serve it in production mode
```bash 
bentoml build service/ 
```
or
```bash
cd service/ # get in 'bentofile.yaml' location 
bentoml build 
bentoml list # to list bento instances different from models list 
```
```bash
bentoml serve iris_classifier:latest --production
```
### Request the service to  make predictions
```bash
python 02_request_service.py 
```

### Serve the model Method 3: containerize the bento with docker
First build a bento as in method 2, then deploy it with docker
```bash
bentoml containerize iris_classifier:tag 
docker images # new image built 
docker run -p 3000:3000 iris_classifier:tag 
```

The service is running on localhost port 3000, we can request for predictions
```bash
python 02_request_service.py 
```

### Summarize: Three major steps for deploying models with BentoML
* Select the model you want to deploy, in this case it was the latest trained model or the latest one having the label stage in production, but you can choose it from a tracking system such as MLflow.
* Save the model to the bentoml store.
* Serve the model.
