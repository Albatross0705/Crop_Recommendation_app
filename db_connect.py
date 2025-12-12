import mysql.connector

# Connect to MySQL Database
db = mysql.connector.connect(
    host="localhost",
    user="cropdb",      # Replace with MySQL username
    password="Albatross@1702",  # Replace with your MySQL password
    database="crop_db"         # Replace with your database name
)

cursor = db.cursor()
cursor.execute("SHOW TABLES")

print("Connected! Available Tables:")
for table in cursor:
    print(table)

db.close()
