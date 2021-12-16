import pyrebase

config = {
"apiKey": "AIzaSyDBiCMJopq-l6ZhMCQJ04OG7Nt26RItt2I",
  "authDomain": "doan-62a4f.firebaseapp.com",
  "databaseURL": "https://doan-62a4f-default-rtdb.firebaseio.com",
  "projectId": "doan-62a4f",
  "storageBucket": "doan-62a4f.appspot.com",
  "messagingSenderId": "237195299576",
  "appId": "1:237195299576:web:182764ba7ec8e491d7490d",
  "measurementId": "G-NP78JXL42Y"
  
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()
data = {"2": ""}
db.push(data)
