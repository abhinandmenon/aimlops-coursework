# pull python base image
FROM python:3.12-slim

# copy application files
ADD /heart_failure_survival_app /heart_failure_survival_app/

# specify working directory
WORKDIR /heart_failure_survival_app

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# expose port for application
EXPOSE 8004

# start fastapi application
CMD ["python", "app/main.py"]
