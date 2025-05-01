import json
import pymysql

# config.json 불러오기
with open("config.json") as f:
    config = json.load(f)["mysql"]

# MySQL 연결
conn = pymysql.connect(
    host=config["host"],
    port=config["port"],
    user=config["user"],
    password=config["password"],
    database=config["database"]
)

try:
    with conn.cursor() as cursor:
        # 현재 사용자 확인
        cursor.execute("SELECT CURRENT_USER();")
        user = cursor.fetchone()[0]
        print(f"Connected as: {user}")

        # 권한 확인
        cursor.execute("SHOW GRANTS;")
        grants = cursor.fetchall()
        print("\n== GRANTS ==")
        for grant in grants:
            print(grant[0])

finally:
    conn.close()
