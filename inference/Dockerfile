FROM python:3.9.5

#Use working directory /app
WORKDIR /app
 
#Copy all the content of current directory to /app
ADD . /app

RUN pip install -r requirements.txt

EXPOSE 5000

#Run python program
CMD ["python","app.py"]
