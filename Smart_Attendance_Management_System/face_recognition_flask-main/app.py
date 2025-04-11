import cv2
import os
from flask import Flask, request, render_template, redirect, url_for
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

app = Flask(__name__)

nimgs = 20
imgBackground = cv2.imread("background.png")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create necessary directories and files
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll Number,Entry Time,Exit Time,Re-Entry Time,Duration (seconds),Period\n')  # Added Duration

# Global variables to track periods
current_period = 1
period_start_time = datetime.now()
period_duration = timedelta(minutes=2)  # 2 minutes per period

# Variables to prevent duplicate attendance
last_user = None
last_time = None

# Dictionary to track user presence
user_presence = {}

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except Exception as e:
        print(f"Error in extract_faces: {e}")
        return []

def identify_face(facearray):
    model = tf.keras.models.load_model('static/face_recognition_model.h5')
    facearray = facearray.reshape(1, 50, 50, 3) / 255.0  # Reshape and normalize
    predictions = model.predict(facearray)
    userlist = os.listdir('static/faces')
    
    if len(userlist) == 0:  # If no users are registered
        print("No registered users found.")
        return None
    
    predicted_user_index = np.argmax(predictions)
    
    if predicted_user_index >= len(userlist):  # Prevent index error
        print("Invalid prediction index:", predicted_user_index)
        return None
    
    return userlist[predicted_user_index]

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    label_dict = {user: i for i, user in enumerate(userlist)}

    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face)
            labels.append(label_dict[user])

    faces = np.array(faces) / 255.0  # Normalize the images
    labels = np.array(labels)
    labels = to_categorical(labels)  # One-hot encode the labels

    # Build the CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(userlist), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(faces, labels, epochs=10, batch_size=32)

    # Save the model
    model.save('static/face_recognition_model.h5')

def extract_attendance():
    try:
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
        
        # Print the DataFrame to debug
        print("DataFrame contents:")
        print(df.head())  # Show the first few rows of the DataFrame
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
        
        # Check if the DataFrame is empty
        if df.empty:
            return [], [], [], [], [], [], [], 0, []  # Return empty lists and zero count if no data

        # Check if the 'Period' column exists
        if 'Period' not in df.columns:
            raise KeyError("The 'Period' column is missing from the attendance file.")

        names = df['Name']
        rolls = df['Roll Number']
        entry_times = df['Entry Time']
        exit_times = df['Exit Time']
        re_entry_times = df['Re-Entry Time']  # New column for re-entry times
        durations = df['Duration (seconds)']  # New column for duration
        periods = df['Period']
        l = len(df)
        
        # Return all necessary values, including re-entry times and durations
        return names, rolls, entry_times, exit_times, re_entry_times, durations, periods, l, entry_times.tolist()  # Return entry_times as a list
    except Exception as e:
        print(f"Error reading attendance file: {e}")
        return [], [], [], [], [], [], [], 0, []  # Return empty lists and zero count on error

def add_attendance(name):
    global current_period, last_user, last_time, user_presence
    print(f"Adding attendance for: {name}")  # Debug statement
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    # Define the start time of the first period (e.g., 11:50:00)
    first_period_start_time = datetime.strptime("22:45:00", "%H:%M:%S").time()

    # Calculate the end time of the last period (period 7)
    last_period_end_time = (datetime.combine(date.today(), first_period_start_time) + timedelta(minutes=14)).time()

    # Get the current time as a time object
    current_time_obj = datetime.now().time()

    # Check if the session has ended
    if current_time_obj > last_period_end_time:
        return "All activities on today’s schedule have been completed"

    # Calculate the difference between the current time and the first period start time
    time_difference = datetime.combine(date.today(), current_time_obj) - datetime.combine(date.today(), first_period_start_time)

    # Calculate the current period based on the time difference (each period is 2 minutes)
    current_period = (time_difference.seconds // 120) + 1  # 120 seconds = 2 minutes

    # Ensure the period does not go below 1
    if current_period < 1:
        current_period = 1

    # Ensure the period does not exceed 7
    if current_period > 7:
        current_period = 7

    print(f"Current period: {current_period}, User ID: {userid}")  # Debug statement

    # Read the CSV file
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')

    # Check if the user is already marked present for the current period
    user_present = df[(df['Roll Number'] == int(userid)) & (df['Period'] == current_period)]

    if not user_present.empty:
        # User is already present, check if they have exited
        if user_present['Exit Time'].iloc[0] is not None:
            # User has exited, record re-entry time
            df.loc[(df['Roll Number'] == int(userid)) & (df['Period'] == current_period), 'Re-Entry Time'] = current_time
            print(f"User    {userid} re-entered at {current_time} for period {current_period}.")
        else:
            print(f"User    {userid} already marked present for period {current_period}.")  # Debug statement
            return
    else:
        # User is not present, add new attendance record
        new_record = pd.DataFrame({
            'Name': [username],
            'Roll Number': [int(userid)],
            'Entry Time': [current_time],
            'Exit Time': [None],  # Initially, exit time is None
            'Re-Entry Time': [None],  # Initially, re-entry time is None
            'Duration (seconds)': [None],  # Initially, duration is None
            'Period': [current_period]
        })

        # Append to the CSV file
        df = pd.concat([df, new_record], ignore_index=True)
        print(f"Attendance recorded for {username} ({userid}) at {current_time} for period {current_period}.")  # Debug statement

    # Save the updated DataFrame back to the CSV file
    df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)

    # Update last user and time
    last_user = userid
    last_time = datetime.now()

    # Update user presence
    user_presence[userid] = {
        'entry_time': current_time,
        'exit_time': None
    }

def update_exit_time(userid):
    global user_presence
    current_time = datetime.now().strftime("%H:%M:%S")
    
    if userid in user_presence:
        user_presence[userid]['exit_time'] = current_time

        # Read the CSV file
        df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')

        # Update the exit time for the user
        user_row = df[(df['Roll Number'] == int(userid)) & (df['Exit Time'].isnull())]

        if not user_row.empty:
            # Update the exit time for the user
            df.loc[user_row.index, 'Exit Time'] = current_time

            # Calculate the duration in seconds
            entry_time = user_row['Entry Time'].iloc[0]
            entry_time_dt = datetime.strptime(entry_time, "%H:%M:%S")
            exit_time_dt = datetime.strptime(current_time, "%H:%M:%S")
            duration = (exit_time_dt - entry_time_dt).total_seconds()  # Duration in seconds

            # Update the duration in the DataFrame
            df.loc[user_row.index, 'Duration (seconds)'] = duration

            # Save the updated DataFrame back to the CSV file
            df.to_csv(f'Attendance/Attendance-{datetoday}.csv', index=False)

            print(f"Exit time updated for user {userid} at {current_time}. Duration: {duration} seconds.")
        else:
            print(f"No entry found for user {userid} to update exit time.")

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l

@app.route('/')
def home():
    names, rolls, entry_times, exit_times, re_entry_times, durations, periods, l, times = extract_attendance()

    # Define the start time of the first period (e.g., 11:50:00)
    first_period_start_time = datetime.strptime("22:45:00", "%H:%M:%S").time()

    # Calculate the end time of the last period (period 7)
    last_period_end_time = (datetime.combine(date.today(), first_period_start_time) + timedelta(minutes=14)).time()

    # Get the current time as a time object
    current_time_obj = datetime.now().time()

    # Check if the session has ended
    session_ended = current_time_obj > last_period_end_time

    return render_template('home.html', names=names, rolls=rolls, entry_times=entry_times, exit_times=exit_times, re_entry_times=re_entry_times, durations=durations, periods=periods, l=l, totalreg=totalreg(), datetoday2=datetoday2, current_period=current_period, session_ended=session_ended, times=times)

@app.route('/start', methods=['GET'])
def start():
    names, rolls, entry_times, exit_times, re_entry_times, durations, periods, l, times = extract_attendance()

    # Define the start time of the first period (e.g., 11:50:00)
    first_period_start_time = datetime.strptime("22:45:00", "%H:%M:%S").time()

    # Calculate the end time of the last period (period 7)
    last_period_end_time = (datetime.combine(date.today(), first_period_start_time) + timedelta(minutes=14)).time()

    # Get the current time as a time object
    current_time_obj = datetime.now().time()

    # Check if the session has ended
    if current_time_obj > last_period_end_time:
        return render_template('home.html', names=names, rolls=rolls, entry_times=entry_times, exit_times=exit_times, re_entry_times=re_entry_times, durations=durations, periods=periods, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess="Today’s agenda has been fully addressed", times=times)

    if 'face_recognition_model.h5' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, entry_times=entry_times, exit_times=exit_times, re_entry_times=re_entry_times, durations=durations, periods=periods, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.', times=times)

    ret = True
    cap = cv2.VideoCapture(0)
    detected_users = set()

    while ret:
        ret, frame = cap.read()
        faces = extract_faces(frame)
        current_detected_users = set()

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))
            if identified_person is not None:
                userid = identified_person.split('_')[1]
                current_detected_users.add(userid)
                if userid not in detected_users:
                    add_attendance(identified_person)
                    detected_users.add(userid)
                cv2.putText(frame, f'{identified_person}', (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        # Check for users who have left the frame
        for userid in detected_users - current_detected_users:
            update_exit_time(userid)
            detected_users.remove(userid)

        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow('Attendance', imgBackground)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, entry_times, exit_times, re_entry_times, durations, periods, l, times = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, entry_times=entry_times, exit_times=exit_times, re_entry_times=re_entry_times, durations=durations, periods=periods, l=l, totalreg=totalreg(), datetoday2=datetoday2, current_period=current_period, times=times)

@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, entry_times, exit_times, re_entry_times, durations, periods, l, times = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, entry_times=entry_times, exit_times=exit_times, re_entry_times=re_entry_times, durations=durations, periods=periods, l=l, totalreg=totalreg(), datetoday2=datetoday2, current_period=current_period, times=times)

if __name__ == '__main__':
    app.run(debug=True)