# pull python base image
FROM python:3.12-slim

# copy application files
ADD /customer_churn_model_api /customer_churn_model_api/

# specify working directory
WORKDIR /customer_churn_model_api

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app/main.py"]
