import mysql.connector as sql
conn=sql.connect(host='localhost',user='root',password='Som_16')
mycursor=conn.cursor()
conn.autocommit = True
mycursor.execute("create database SIH")
mycursor.execute("use SIH")
mycursor.execute("create table Result_Data(Plant_Name Varchar(50) Primary Key, Information  text(5000), Geographical_Loaction text(5000), Climatic_Condition text(5000), Medical_Benefits text(5000))")