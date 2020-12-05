import sqlite3
connection = sqlite3.connect("compas.db")
cursor = connection.cursor()
cursor.execute("SELECT * FROM charge;")
names = [description[0] for description in cursor.description]
print(names)


cursor = connection.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())
