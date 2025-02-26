# Floral Area Estimation Web Application

This repository contains a web application for estimating floral area using AI-powered algorithm. The application is built with YOLOv8 and integrates with **Gradio** for an interactive user interface.

![alt text](image.png)

## Features
- **AI-based floral area estimation**
- **Easy-to-use web interface** powered by Gradio
- **Run locally with Gradio or via Docker**

---

## 🚀 Getting Started  

You can run this software in **two ways**:  

1️⃣ **Clone the repository and run it locally**  
2️⃣ **Use Docker to run a pre-built image**  

---

## 🛠️ Running Locally  

Follow these steps to set up and run the application on your local machine:

### **1️⃣ Clone the repository**
```
git clone https://github.com/eai6/FloralArea_Web.git
cd FloralArea_Web
```

### Create a virtual environment  
```
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### Install dependencies
```
pip install -r requirements.txt
```

### Run the application
```
python app.py
```

## Running with Docker
If you prefer running the application using Docker, follow these steps:

### Pull the docker image
```
docker pull eai6/floralarea:x86
```

### Run the docker container
```
docker run -p 7860:7860 eai6/floralarea:x86
```

📚 Edward I. Amoah<sup>1,3</sup>, Khayri White<sup>2</sup>, Harland M. Patch<sup>3</sup>, and Christina M. Grozinger<sup>4</sup>

1️⃣ Edward I. Amoah
Intercollege Graduate Degree Program in Ecology, Huck Institutes of the Life Sciences, Penn State University, University Park, PA

2️⃣ Khayri White
Undergraduate Degree Program in Computer Engineering, Department of Electrical Engineering and Computer Science, Howard University, Washington, DC

3️⃣ Harland M. Patch
Department of Entomology, Center for Pollinator Research, Huck Institutes of the Life Sciences, Penn State University, University Park, PA

4️⃣ Christina M. Grozinger
Department of Entomology, Center for Pollinator Research, Huck Institutes of the Life Sciences, Penn State University, University Park, PA