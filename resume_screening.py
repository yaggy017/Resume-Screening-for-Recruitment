# main.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# 🔹 Sample resumes and job roles
resumes = [
    "I have experience in manual testing, automation testing using Selenium, writing test cases, bug reporting and working with QA team.",
    "I am skilled in UI/UX design, Figma, Adobe XD, wireframing, prototyping, user research and design thinking.",
    "Proficient in Python, pandas, machine learning, deep learning and data science.",
    "Worked on backend development using Java, Spring Boot, and REST APIs.",
    "Experienced in data entry, Excel, Google Sheets, typing and admin tasks.",
    "Strong knowledge of HTML, CSS, JavaScript, and React for front-end development.",
    "Familiar with Figma, prototyping, design tools, and UX research.",
    "Worked with routers, switches, servers, and IT infrastructure support.",
    "Analyzed data using Power BI, Excel dashboards, and Tableau.",
    "Managed HR tasks, recruitment, payroll, and employee records.",
    "Knowledge of accounting principles, Tally, GST filing, and bookkeeping.",
    "Good at communication, leadership, presentations, and public speaking.",
    "Built Android apps using Java and Kotlin, familiar with Android Studio.",
    "Developed iOS apps using Swift, Xcode and deployed to App Store.",
    "Designed graphics using Photoshop, Illustrator and Canva.",
    "Created video content, edited with Premiere Pro and After Effects.",
    "Worked with IoT devices, sensors, Arduino and Raspberry Pi.",
    "Skilled in AWS, EC2, S3, Lambda and cloud deployment.",
    "Experience in cybersecurity, ethical hacking and penetration testing.",
    "Familiar with database management using MySQL, MongoDB and SQL queries.",
    "Worked as school teacher handling maths and science subjects.",
    "Trained students in soft skills, personality development and spoken English.",
    "Experience in customer support, handling queries and service tickets.",
    "Worked in BPO handling inbound and outbound customer calls."
]

labels = [
    "QA Tester", "UI/UX Designer", "Data Scientist", "Backend Developer",
    "Data Entry Operator", "Frontend Developer", "UI/UX Designer",
    "Network Engineer", "Data Analyst", "HR Executive", "Accountant",
    "Soft Skills Trainer", "Android Developer", "iOS Developer",
    "Graphic Designer", "Video Editor", "IoT Engineer", "Cloud Engineer",
    "Cybersecurity Analyst", "Database Administrator", "School Teacher",
    "Soft Skills Trainer", "Customer Support", "BPO Executive"
]

# Step 1: Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(resumes)

# Step 2: Train SVM
model = LinearSVC()
model.fit(X, labels)

# Step 3: User input
print("🔹 Enter your resume content below (skills, experience, etc.):")
user_resume = input("📝 Your Resume: ")

# Step 4: Predict
user_vector = vectorizer.transform([user_resume])
predicted_role = model.predict(user_vector)

# Step 5: Output
print(f"\n✅ Based on your resume, you are suitable for the role of: **{predicted_role[0]}**")
