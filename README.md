# 💊 Pillcare: Smart Medicine Information and Reminder Platform

Pillcare is a smart medicine information and reminder platform developed as an academic project.

It helps users maintain a simple medicine schedule, track daily doses, view dose history, monitor medicine supply, and look up medicine terminology using a reference source.

## ✨ Features

- Add medicines with name, dose, scheduled time, and supply information
- Edit existing medicine details
- Delete medicines
- Search medicines in the personal medicine list
- Track today's scheduled doses
- Mark doses as Taken
- Store the date and time when a dose was taken
- View Dose History
- Show Due Now, Overdue, and Upcoming reminder status
- Show the next upcoming medicine reminder
- Create today's and tomorrow's dose occurrences
- Monitor medicine supply and identify medicines that need refilling
- Medicine terminology lookup using RxNorm
- Local data persistence using SQLite
- Preserve historical medicine name and dose for newly created dose records

## 🛠️ Technologies Used

- Python 3.11
- Streamlit
- SQLite
- SQLAlchemy
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Requests
- python-dotenv
- RxNorm

## 🏗️ Project Architecture

```text
User
  ↓
Streamlit Interface
  ↓
Application Logic
  ├── Medicine Management
  ├── Dose Scheduling
  ├── Dose Tracking
  ├── Reminder Status
  ├── Refill Tracking
  └── Medicine Information Lookup
  ↓
SQLite Database