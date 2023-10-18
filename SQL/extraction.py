import mysql.connector as sql
def Find(result):
    conn=sql.connect(host='localhost',user='root',password='Som_16',database='Sih')
    mycursor=conn.cursor()
    conn.autocommit = True
    result = [result]
    mycursor.execute('select * from result_data where Plant_name=(%s)',(result))
    data=mycursor.fetchall()
    a=[]
    for i in data:
        a.append(i)
                                
    if len(a)!=1:
        print('~!~!~!~!~~NO DATA FOUND~~!~!~!~!~')
        return ''
                                
    else:
        print(a)
        return a

Find('aloe vera')