import cx_Oracle
import os

username = 'chemdb'
password = 'chem4db'
dsn = 'virusdb.camd.krict.re.kr/orcl'
port = 1521
endoding = 'UTF-8'

# os.environ["ORACLE_HOME"] = "/opt/oracle/instantclient_21_9"

try:
    connection = cx_Oracle.connect( username, password, dsn, encoding=endoding )
except Exception as ex :
    print( 'Could not connect to database :', ex )


try:
    sql = "select * from PI"
    cursor = connection.cursor()
    cursor.execute( sql )

    rows = cursor.fetchall()
    print( rows )

    cursor.close()
    connection.close()

except Exception as ex:
    print( 'Sql error: ', ex )

