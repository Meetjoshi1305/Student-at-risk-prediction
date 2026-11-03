# Student-at-risk-prediction
Student Result Early-Warning System is a machine learning mini-project. It predicts whether a student is likely to pass ,fail or may need extra help.

Student Result Early-Warning System: full explanation

1. The idea in one line
It looks at a student’s attendance, assignment marks and previous exam marks, compares them with past students, and says whether the student will probably pass or needs extra help.

2. The problem it solves
Teachers usually find out a student is failing only after the exam. This project warns them earlier, so they can help in time.

3. The data
   -What: 800 students in students.csv. It is synthetic, meaning computer-generated, because real student records are private.
   -Columns: student_id, attendance (%), assignment_marks (/100), previous_exam_marks (/100) and passed (1 = pass, 0 = fail).
   -How pass/fail was decided: each student gets a score of 30% attendance + 30% assignments + 40% previous exam, plus random noise. A higher score makes passing             more likely. Because of the noise, the pattern is real but not perfect, like real life.
   -Split: 425 passed (53%) and 375 failed (47%).

4. The algorithm: XGBoost
   -A decision tree asks simple yes/no questions, such as “attendance above 60%?”.
   -Boosting builds many small trees one after another. Each new tree focuses on the students the earlier trees got wrong.
   -XGBoost is a fast, regularised version of this. Regularised means it is restrained from memorising the data.
   -Settings: 150 trees, depth 3 (each tree asks at most 3 questions in a row), learning rate 0.1 (each tree adds only a small correction).
   -Output: a pass probability from 0 to 1.

5. Steps of the project
   -Generate or load the data.
   -Check that the columns exist and both outcomes are present.
   -Split 80% for training (640 students) and 20% for testing (160 students). The split is stratified, so both parts keep the same pass/fail ratio.
   -Train XGBoost on the 640 training students.
   -Test it on the 160 students it has never seen.
   -Use it in the app to predict for a new student.

6. The app (Streamlit)
  Tab 1, Past data: shows the table, pass rate, averages and a chart.
  Tab 2, Train model: sliders for trees, depth and learning rate, then a Train button. It shows accuracy, the confusion matrix and feature importance.
  Tab 3, Predict: enter a student’s three values and get the probability.
  70% or more: Likely to pass.
  40% to 70%: Borderline.
  Below 40%: At risk.
  You can also upload your own CSV with the same columns.

7. Limitations
  -The data is synthetic, so the results show that the method works, not how it would do on real students.
  -The test set is small (160 students).
  -It is an estimate, not a guarantee.
  -The 70% and 40% cut-offs are my own choice and do not come from the model.
  -A simple logistic regression scored 78.1%, higher than XGBoost’s 73.8%. This is because I generated the data from a nearly straight-line formula. On real data         with more complex patterns, XGBoost is expected to do better.

8. Files
  app.py: the web app and model.
  generate_data.py: creates the dataset.
  students.csv: the data.
  requirements.txt: the libraries to install.
  README.md: how to run it.
Student_Early_Warning_MiniProject_Report.docx: the 13-page report.

To run: pip install -r requirements.txt, then streamlit run app.py.
