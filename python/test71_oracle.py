import oracledb


with oracledb.connect( user="SCOTT", password="TIGER", dsn="10.13.22.178/orcl") as conn :
    with conn.cursor() as cursor :
        sql = "select table_name from user_tables"
        sql = "select ENAME, JOB, MGR, HIREDATE, DNAME from EMP, DEPT" \
              " where EMP.DEPTNO = DEPT.DEPTNO"
        for result in cursor.execute( sql ) :
            print(result)





import requests
from bs4 import BeautifulSoup

with requests.get( "https://mis.krict.re.kr" ) as webpage :
    soup = BeautifulSoup( webpage.content, "html.parser" )
    title_list = soup.find_all( 'span', class_ = 'title' )
    for span in title_list:
        div = span.select( 'div > span' )
        for l in span:
            print( l.get_text() )



