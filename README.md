

\# AI-Based Student Placement Prediction System



\## Overview

The AI-Based Student Placement Prediction System is a machine-learning project designed to estimate student placement outcomes using academic performance, skills, internship experience, and project experience.



\## Features

\- Student placement prediction using a saved machine-learning model

\- Placement analytics dashboard with interactive charts

\- Career Advisor with skill-gap recommendations

\- Batch prediction using CSV uploads

\- Downloadable placement prediction results

\- PDF student assessment reports

\- Model performance and feature-importance analytics



\## Technology Stack

\- \*\*Language:\*\* Python

\- \*\*Frontend:\*\* Streamlit

\- \*\*Data Processing:\*\* Pandas, NumPy

\- \*\*Machine Learning:\*\* Scikit-learn

\- \*\*Visualization:\*\* Plotly, Matplotlib, Seaborn

\- \*\*Model Storage:\*\* Joblib

\- \*\*PDF Reports:\*\* ReportLab



\## Project Structure

```text

AI-Student-Placement-Prediction/

├── app.py

├── requirements.txt

├── README.md

├── college\_student\_placement\_dataset.csv

├── final\_random\_forest\_model.pkl

└── pages/

&#x20;   ├── 1\_Dashboard.py

&#x20;   ├── 3\_Career\_Advisor.py

&#x20;   ├── 4\_Batch\_Prediction.py

&#x20;   ├── 5\_Model\_Analytics.py

&#x20;   └── 6\_Student\_Reports.py

```



\## Installation



1\. Clone the repository:

&#x20;  ```bash

&#x20;  git clone https://github.com/Aditya13011/AI-Student-Placement-Prediction.git

&#x20;  ```



2\. Move into the project directory:

&#x20;  ```bash

&#x20;  cd AI-Student-Placement-Prediction

&#x20;  ```



3\. Create a virtual environment:

&#x20;  ```bash

&#x20;  python -m venv venv

&#x20;  ```



4\. Activate it on Windows PowerShell:

&#x20;  ```powershell

&#x20;  .\\venv\\Scripts\\Activate.ps1

&#x20;  ```



5\. Install dependencies:

&#x20;  ```bash

&#x20;  python -m pip install -r requirements.txt

&#x20;  ```



6\. Start the application:

&#x20;  ```bash

&#x20;  python -m streamlit run app.py

&#x20;  ```



\## Dataset Features

The application uses student-related attributes such as:

\- IQ

\- Previous semester result

\- CGPA

\- Academic performance

\- Internship experience

\- Extra-curricular score

\- Communication skills

\- Number of completed projects



\## Disclaimer

Predictions are model estimates for educational demonstration. They do not guarantee employment or placement. Model performance must be evaluated on suitable unseen data before being interpreted as evidence of real-world accuracy.



