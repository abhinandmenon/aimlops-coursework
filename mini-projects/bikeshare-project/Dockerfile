# pull python base image
FROM python:3.12-slim

# copy application files
ADD /bikeshare_model_api /bikeshare_model_api/

# specify working directory
WORKDIR /bikeshare_model_api

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt
RUN rm *.whl

# expose port for application
EXPOSE 8002

# start fastapi application
CMD ["python", "app/main.py"]
