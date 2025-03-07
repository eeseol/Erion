import os
import sys
import json
import pymysql
import re

from PyQt5.QtWidgets import *
from PyQt5.QtCore import QEvent, Qt, QTimer, QObject, pyqtSignal
from PyQt5 import uic

from random import uniform, normalvariate, randint
import time
import paho.mqtt.client as mqtt
import paho.mqtt.subscribe as subscribe
import RPi.GPIO as GPIO

def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)
    
# main 화면
form = resource_path('main.ui')
form_class = uic.loadUiType(form)[0]

# 고객용 화면
form_second = resource_path('customer_mode.ui')
form_secondwindow = uic.loadUiType(form_second)[0]

# 관리자용 화면
form_third = resource_path('admin_mode.ui')
form_thirdwindow = uic.loadUiType(form_third)[0]

# 상품 관리용 화면
form_fourth = resource_path('product_managing_mode.ui')
form_fourthwindow = uic.loadUiType(form_fourth)[0]

GPIO_PIN_17 = 17
GPIO_PIN_27 = 27
GPIO_PIN_22 = 22

screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption('MQTT UI Example')
font = pygame.font.SysFont(None, 55)

#mqtt 클라이언트 클래스 정의
class MQTTClient:
    def __init__(self):
        #mqtt 클라이언트 생성
        self.client = mqtt.Client()
        #mqtt 클라이언트의 이벤트 콜백 설정
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        #mqtt 브로커 연결
        self.client.connect("브로커 주소",1883)
        #비동기적으로 네트워크 루프 시작
        self.client.loop_start()
   
    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        self.client.subscribe("tuk/test/topic")
       
    def on_message(self, client, userdata, msg):
        print(msg.topic + " " + str(msg.payload))
        main_window.display_message(msg.payload.decode())
       
    def publish(self, topic, message):
        self.client.publish(topic, message)

class Database:

    # product data 전부 가져오는 메서드
    @staticmethod
    def product_data():
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='raspberry',
            database='test'
        )
       
        try:
            with connection.cursor() as cursor:
                # SQL 쿼리
                sql = "SELECT BARCODE, PRONAME, PRICE FROM product"
                cursor.execute(sql)
                result = cursor.fetchall()

                return result
        finally:
            connection.close()
            
    # 바코드를 통해 product에서 탐색하고 정보를 반환하는 메서드
    @staticmethod
    def product_find(barcode):
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='raspberry',
            database='test'
        )     
        try:
            with connection.cursor() as cursor:
                                # 바코드에 해당하는 상품 정보 조회
                sql = "SELECT PRONAME, PRICE FROM product WHERE BARCODE = %s"
                cursor.execute(sql, (barcode,))
                product_info = cursor.fetchone()

                if product_info:    #바코드 정보가 product에 있을 경우
                    proname = product_info[0]
                    price = product_info[1]
                else:               #바코드 정보가 product에 없는 경우
                    proname = 'none'
                    price = 'none'
                    print("해당하는 상품이 존재하지 않습니다.")
                
                return barcode, proname, price
        finally:
            connection.close()   
    
    #product 데이터를 업데이트하는 메서드
    @staticmethod
    def product_fix(old_barcode, new_barcode, new_proname, new_price):
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='raspberry',
            database='test'
        )
        try:
            with connection.cursor() as cursor:
                sql_update = "UPDATE product SET BARCODE = %s, PRONAME = %s, PRICE = %s WHERE BARCODE = %s"
                cursor.execute(sql_update, (new_barcode, new_proname, new_price, old_barcode))
                connection.commit()
        except Exception as e:
            connection.rollback()  # 롤백 처리
        finally:
            connection.close()
            
    #product에 데이터를 추가하는 메서드        
    @staticmethod
    def product_add(barcode, proname, price):
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='raspberry',
            database='test'
        )        
        try:
            with connection.cursor() as cursor:
                # SQL 쿼리
                sql = "INSERT INTO product (BARCODE, PRONAME, PRICE) VALUES (%s, %s, %s)"
                cursor.execute(sql, (barcode, proname, price))
                result = cursor.fetchall()
                connection.commit()
        finally:
            connection.close()
    
    #product에서 데이터를 지우는 메서드 
    @staticmethod
    def product_delete(barcode):
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='raspberry',
            database='test'
        )
        try:
            with connection.cursor() as cursor:
                # SQL 쿼리
                sql = "DELETE FROM product WHERE BARCODE = %s"
                cursor.execute(sql, (barcode))
                result = cursor.fetchall()
                connection.commit()
        finally:
            connection.close()

    @staticmethod
    def shopping_data ():
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='raspberry',
            database='test'
        )
        try:
            with connection.cursor() as cursor:
                sql = "SELECT BARCODE, PRONAME, AMOUNT, TOTAL_PRICE FROM shopping"
                cursor.execute(sql)
                result = cursor.fetchall()
                
                sql = "SELECT SUM(TOTAL_PRICE) AS total_sales FROM shopping"
                cursor.execute(sql)
                total_price = cursor.fetchall()
                
                sql = "SELECT SUM(AMOUNT) AS total_amount FROM shopping"
                cursor.execute(sql)
                total_ammount = cursor.fetchall()              
                
                
                return result, total_price, total_ammount
        finally:
            connection.close()

    @staticmethod
    def shopping_find_amount(barcode):
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='raspberry',
            database='test'
        )	
        try:
            with connection.cursor() as cursor:
                sql = "SELECT AMOUNT FROM shopping WHERE BARCODE = %s"
                cursor.execute(sql, (barcode,))
                result = cursor.fetchall()             
                
                return result[0][0]
        finally:
            connection.close()

    @staticmethod
    def shopping_plus_amount(barcode, amount):
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='raspberry',
            database='test'
        )	
        try:
            with connection.cursor() as cursor:
                
                new_amount = amount + 1
                sql = "UPDATE shopping SET AMOUNT = %s WHERE BARCODE = %s"
                cursor.execute(sql, (new_amount, barcode,))
                result = cursor.fetchall()             
                
                connection.commit()
                
        finally:
            connection.close()

    @staticmethod
    def shopping_minus_amount(barcode, amount):
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='raspberry',
            database='test'
        )	
        try:
            with connection.cursor() as cursor:
                
                new_amount = amount - 1
                sql = "UPDATE shopping SET AMOUNT = %s WHERE BARCODE = %s"
                cursor.execute(sql, (new_amount, barcode,))
                result = cursor.fetchall()             
                
                connection.commit()
                
        finally:
            connection.close()

    @staticmethod
    def shopping_delete_data(barcode):
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='raspberry',
            database='test'
        )	
        try:
            with connection.cursor() as cursor:
                
                sql = "DELETE FROM shopping WHERE BARCODE = %s"
                cursor.execute(sql, (barcode,))
                result = cursor.fetchall()             
                
                connection.commit()
                
        finally:
            connection.close()
            
    @staticmethod
    def shopping_plus_data(barcode, proname, price):
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='raspberry',
            database='test'
        )
        try:
            with connection.cursor() as cursor:

                try:
                    price = float(price)  # 가격을 숫자로 변환 시도
                except ValueError:
                    price = 0  # 기본 가격 설정
                
                sql = "INSERT INTO shopping (BARCODE, PRONAME, PRICE, AMOUNT) VALUES (%s, %s, %s, 1)"
                cursor.execute(sql, (barcode, proname, price))
                result = cursor.fetchall()             
                
                connection.commit()
                
        finally:
            connection.close()
        
            
    # 바코드를 통해 product에서 탐색하고 정보를 반환하는 메서드
    @staticmethod
    def shopping_find(barcode):
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='raspberry',
            database='test'
        )     
        try:
            with connection.cursor() as cursor:
                                # 바코드에 해당하는 상품 정보 조회
                sql = "SELECT PRONAME, PRICE FROM shopping WHERE BARCODE = %s"
                cursor.execute(sql, (barcode,))
                product_info = cursor.fetchone()

                if product_info is not None:    # 바코드 정보가 product에 있을 경우
                    return True
                else:                          # 바코드 정보가 product에 없는 경우
                    return False
                
        finally:
            connection.close()

    @staticmethod
    def shopping_clear_data():
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='raspberry',
            database='test'
        ) 
        try:
            with connection.cursor() as cursor:
                                # 바코드에 해당하는 상품 정보 조회
                sql = "DELETE FROM shopping"
                cursor.execute(sql)
                product_info = cursor.fetchone()
                
                connection.commit()
            
        finally:
            connection.close()        
        
        
# main 화면 클래스 정의
class MainWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.mqtt_client = MQTTClient()
        
        GPIO.setmode(GPIO.BCM)  # BCM 모드로 GPIO 모드 설정
        GPIO.setup(GPIO_PIN_17, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(GPIO_PIN_27, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        GPIO.setup(GPIO_PIN_22, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.checkGPIO)
        self.timer.start(100)  # 0.1초마다 GPIO 상태 체크
        
        
        self.pushButton.clicked.connect(self.go_to_customer_mode)
        self.pushButton_2.clicked.connect(self.go_to_admin_mode)

    def go_to_customer_mode(self): #고객용 버튼
        self.mqtt_client.publish("tuk/mode", "cus")
        self.hide()  # 첫 번째 화면 숨기기
        self.second_window = SecondWindow(self.mqtt_client)
        self.second_window.show()

    def go_to_admin_mode(self): #관리자용 버튼
        self.mqtt_client.publish("tuk/mode", "admin")
        self.hide()
        self.third_window = ThirdWindow(self.mqtt_client)
        self.third_window.show()    
   
    def display_message(self, message):
        self.label.setText(message)
        
    def checkGPIO(self):
        if GPIO.input(GPIO_PIN_17) == GPIO.HIGH:
            if GPIO.input(GPIO_PIN_27) == GPIO.HIGH:
                msgBox = QMessageBox(self)
                msgBox.setWindowTitle('충돌 감지됨!')
                msgBox.setText('3단계 충돌 위험 신호가 감지되었습니다.')
                msgBox.setIcon(QMessageBox.Critical)
                msgBox.setStandardButtons(QMessageBox.Ok)
                msgBox.exec_()
    
    def closeEvent(self, event):
        GPIO.cleanup()
        event.accept()

# 고객용 화면 클래스 정의
class SecondWindow(QDialog, form_secondwindow):
    def __init__(self, mqtt_client):  # Accept mqtt_client as a parameter
        super().__init__()
        self.setupUi(self)		
        self.load_data()
        self.mqtt_client = mqtt_client    
        self.listWidget.itemClicked.connect(self.on_item_clicked)
        self.accumulated_number = ""

        # QTimer 인스턴스 생성
        self.timer = QTimer(self)
        # QTimer의 timeout 시그널이 발생할 때 self.select_total_price_view 호출
        self.timer.timeout.connect(self.select_total_price_view)
        # 타이머 간격을 500ms(0.5초)로 설정
        self.timer.start(500)
        
        self.pushButton_call.setStyleSheet("background-color: red")
        
    def showEvent(self, event):
        super().showEvent(event)
        # 이 부분에서 total_price_view를 선택하게 만듭니다.
        self.total_price_view.setFocus()  # total_price_view 위젯에 포커스 설정
        self.pushButton_call.setStyleSheet("background-color: red")

        
    def select_total_price_view(self):
        # 타이머가 만료될 때마다 total_price_view를 선택하도록 합니다.
        self.total_price_view.setFocus() 
        
        if GPIO.input(GPIO_PIN_22) == GPIO.HIGH:
            self.mode_view.setPlainText("쇼핑 모드입니다.")  # total_price는 문자열 형태로 처리
        else:
            self.mode_view.setPlainText("트래킹 모드입니다.")  # total_price는 문자열 형태로 처리

        
    def load_data(self):
        
        data, total_price, total_amount = Database.shopping_data()
        
        header_text = " 수량\t가격\t제품명"
        line_text = '------------------------------------------------------------------------------------------------'
        self.listWidget.addItem(header_text)
        self.listWidget.addItem(line_text)

        
        for row in data:
            display_text = f" {row[2]}개\t{row[3]}원\t({row[0]}){row[1]}"  # 수정된 부분: row의 각 요소는 순서대로 접근
            self.listWidget.addItem(display_text)
        
        self.total_price_view.setPlainText(str(total_price[0][0]))  # total_price는 문자열 형태로 처리

        self.total_count_view.setPlainText(str(total_amount[0][0]))  # total_price는 문자열 형태로 처리
    
    def plus_product(self):
        if self.chose_barcode is not None:
            # 데이터베이스에서 선택된 아이템의 수량을 +1 업데이트
            Database.shopping_plus_amount(self.chose_barcode, self.chose_amount)
            # 리스트 위젯 업데이트
            self.listWidget.clear()  # 기존 리스트 초기화
            self.load_data()  # 새로 데이터 로드        
    
    def minus_product(self):
        if self.chose_barcode is not None:      
            if self.chose_amount == 1:
                Database.shopping_delete_data(self.chose_barcode)
            elif self.chose_amount > 1:
                Database.shopping_minus_amount(self.chose_barcode, self.chose_amount)
            else:
                print('error')
      
            # 리스트 위젯 업데이트
            self.listWidget.clear()  # 기존 리스트 초기화
            self.load_data()  # 새로 데이터 로드  
        
    def delet_product(self):
        if self.chose_barcode is not None:
            # 데이터베이스에서 선택된 아이템의 수량을 삭제 업데이트
            Database.shopping_delete_data(self.chose_barcode)

            # 리스트 위젯 업데이트
            self.listWidget.clear()  # 기존 리스트 초기화
            self.load_data()  # 새로 데이터 로드    

        #listWidget 항목 선택 메서드        
    def on_item_clicked(self, item):
                    # 선택된 항목의 텍스트를 \t 기준으로 파싱
        selected_text = item.text()
        
        
            # 헤더나 구분선인 경우 아무 동작도 하지 않음
        if selected_text.startswith('\t상품명') or selected_text.startswith('------------------------------------------------------------------------------------------------'):
            return
                    # 바코드 추출 패턴
        barcode_pattern = r'\((\d+)\)'

        # 바코드 추출
        barcode_match = re.search(barcode_pattern, selected_text)
        if barcode_match:
            self.chose_barcode = barcode_match.group(1)
        else:
            self.chose_barcode = None
        
        #수량 추출
        self.chose_amount = Database.shopping_find_amount(self.chose_barcode)
        
        print(self.chose_barcode)
        print(self.chose_amount)
    
 
    def keyPressEvent(self, e):
        key = e.key()
        if key == Qt.Key_Enter or key == Qt.Key_Return:
            # Enter 키가 눌렸을 때는 저장된 숫자를 출력하고 변수를 초기화합니다.
            #self.barcode_output.append(self.accumulated_number)
            print(self.accumulated_number)
            
            if Database.shopping_find(self.accumulated_number):
                        # 데이터베이스에서 선택된 아이템의 수량을 +1 업데이트
                self.chose_amount = Database.shopping_find_amount(self.accumulated_number)
                Database.shopping_plus_amount(self.accumulated_number, self.chose_amount)
            else:
                find_barcode, find_proname, find_price = Database.product_find(self.accumulated_number)
                                    
                if find_proname == 'none':    #바코드 정보가 없을때
                    QMessageBox.information(self, '입력 오류', '바코드 정보가 없습니다. \n다시 입력해주세요')
                else:                           #바코드 정보가 이미 존재할때
                    Database.shopping_plus_data(self.accumulated_number, find_proname, find_price)
                
                
            self.accumulated_number = ""


            self.listWidget.clear()  # 리스트 위젯 초기화
            self.load_data()  # 데이터 다시 로드
        elif Qt.Key_0 <= key <= Qt.Key_9:  # 숫자 키 입력인 경우
            # 숫자 키가 눌렸을 때는 입력된 숫자를 변수에 누적 저장합니다.
            self.accumulated_number += e.text()
    
    def call(self):
        QMessageBox.information(self, "직원호출", "직원이 호출되었습니다.")

    def payment(self):
        
        Database.shopping_clear_data()
        reply = QMessageBox.information(self, "Payment", "결제하시겠습니까?", QMessageBox.No | QMessageBox.Yes)
        
        if reply == QMessageBox.Yes:
            QMessageBox.information(self, "Payment check", "결제 성공. 감사합니다.")
            self.mqtt_client.publish("tuk/size", 5)
            
            self.close()  # 두 번째 화면 닫기
            self.mqtt_client.publish("tuk/mode", "main")
            main_window.show()  # 첫 번째 화면 보이기

    def go_to_main(self): #뒤로가기 버튼
        
        Database.shopping_clear_data()

        self.close()  # 두 번째 화면 닫기
        self.mqtt_client.publish("tuk/mode", "main")
        main_window.show()  # 첫 번째 화면 보이기

# 관리자용 화면 클래스 정의            
class ThirdWindow(QDialog, form_thirdwindow):
    def __init__(self, mqtt_client):
        super().__init__()
        self.setupUi(self)
        self.mqtt_client = mqtt_client    

        self.i = 0  # 버튼이 눌렸는지 여부를 나타내는 변수


        self.pushButton_forward.pressed.connect(self.forward_pressed)
        self.pushButton_forward.released.connect(self.stop)

        self.pushButton_forward_left.pressed.connect(self.forward_left_pressed)
        self.pushButton_forward_left.released.connect(self.stop)

        self.pushButton_forward_right.pressed.connect(self.forward_right_pressed)
        self.pushButton_forward_right.released.connect(self.stop)

        self.pushButton_back.pressed.connect(self.back_pressed)
        self.pushButton_back.released.connect(self.stop)

        self.pushButton_back_left.pressed.connect(self.back_left_pressed)
        self.pushButton_back_left.released.connect(self.stop)

        self.pushButton_back_right.pressed.connect(self.back_right_pressed)
        self.pushButton_back_right.released.connect(self.stop)

        self.pushButton_stop.pressed.connect(self.stop)

       
        self.btngroup1 = QButtonGroup()
       
        self.btngroup1.addButton(self.radioButton_screen)
        self.btngroup1.addButton(self.radioButton_joystick)
       
        self.radioButton_screen.setChecked(True)
        
   
    # 리니어 엑추에이터 관련 함수
    def size_0(self):
        self.mqtt_client.publish("tuk/size",0)
    
    def size_1(self):
        self.mqtt_client.publish("tuk/size", 1)
   
    def size_2(self):
        self.mqtt_client.publish("tuk/size", 2)
    def size_3(self):
        self.mqtt_client.publish("tuk/size", 3)
       
    def size_4(self):
        self.mqtt_client.publish("tuk/size", 4)
       
    # 화면 전환 관련 함수
    def go_to_product_management(self):
        self.close()
        self.fourth_window = FourthWindow(self.mqtt_client)  # 수정: FourthWindow 객체 생성 시 mqtt_client 전달
        self.fourth_window.show()
        
    def go_to_main(self):
        self.close()
        self.mqtt_client.publish("tuk/mode", "main")
        main_window.show()  # Show the main wind

       
     # DC 모터 관련 함수    
    def forward_pressed(self):
        self.i = 1
        self.timer.start(500)  # 500ms마다 timeout 신호 발생

    def forward_left_pressed(self):
        self.i = 2
        self.timer.start(500)  # 500ms마다 timeout 신호 발생

    def forward_right_pressed(self):
        self.i = 3
        self.timer.start(500)  # 500ms마다 timeout 신호 발생

    def back_pressed(self):
        self.i = 4
        self.timer.start(500)  # 500ms마다 timeout 신호 발생

    def back_left_pressed(self):
        self.i = 5
        self.timer.start(500)  # 500ms마다 timeout 신호 발생

    def back_right_pressed(self):
        self.i = 6
        self.timer.start(500)  # 500ms마다 timeout 신호 발생

    def stop(self):
        self.i = 0
        self.timer.stop()
        self.mqtt_client.publish("tuk/motor", json.dumps([0, 0, 0, 0]))

    def send_value(self):
        if self.i ==1:
            self.mqtt_client.publish("tuk/motor", json.dumps([0, 0, 1, 0]))
        elif self.i ==2:
            self.mqtt_client.publish("tuk/motor", json.dumps([0, 0, 0, 1]))
        elif self.i ==3:
            self.mqtt_client.publish("tuk/motor", json.dumps([0, 0, 1, 1]))
        elif self.i ==4:
            self.mqtt_client.publish("tuk/motor", json.dumps([0, 1, 0, 1]))
        elif self.i ==5:
            self.mqtt_client.publish("tuk/motor", json.dumps([0, 1, 0, 0]))
        elif self.i ==6:
            self.mqtt_client.publish("tuk/motor", json.dumps([0, 1, 1, 0]))
        else:
            self.mqtt_client.publish("tuk/motor", json.dumps([0, 0, 0, 0]))
        
    # DC 모터 제어 방식 관련 함수
    def screen_control(self):
        screen = self.sender()
        if screen.isChecked():
            self.mqtt_client.publish("tuk/control", "screen")    
       
    def joystick_control(self):
        joystick = self.sender()
        if joystick.isChecked():
            self.mqtt_client.publish("tuk/control", "joystick")
    
    
        
# 상품 관리용 화면 클래스 정의
class FourthWindow(QDialog, form_fourthwindow):
    def __init__(self, mqtt_client):  # Accept mqtt_client as a parameter
        super().__init__()
        self.setupUi(self)
        self.mqtt_client = mqtt_client  # Assign mqtt_client to self.mqtt_client
        
                # 읽기 전용 그룹으로 묶기
        self.readonly_group = [self.chose_barcode, self.chose_proname, self.chose_price]
        
                # 쓰기 전용 그룹으로 묶기
        self.editable_group = [self.fix_barcode, self.fix_proname, self.fix_price]
        
                # 라디오 버튼과 그에 따른 함수를 연결합니다.
        self.radioButton_fix.toggled.connect(self.toggle_fix_mode)          #수정하기 모드
        self.radioButton_add.toggled.connect(self.toggle_add_mode)          #추가하기 모드
        self.radioButton_delete.toggled.connect(self.toggle_delete_mode)    #삭제하기 모드
        
                # 초기에는 '수정' 모드가 선택되도록 합니다.
        self.radioButton_fix.setChecked(True)
        self.toggle_fix_mode()
        
        self.load_data()
        
                # listWidget에서 항목을 선택할 때 발생하는 이벤트를 처리합니다.
        self.listWidget.itemClicked.connect(self.on_item_clicked)

    #데이터들을 listWidget에 띄워주는 메서드
    def load_data(self):
        data = Database.product_data()
        
        header_text = "   바코드\t\t가격\t\t이름"
        line_text = '------------------------------------------------------------------------------------------------'
        self.listWidget.addItem(header_text)
        self.listWidget.addItem(line_text)

        for row in data:
            display_text = f"{row[0]}\t{row[2]}원\t{row[1]}"  # 수정된 부분: row의 각 요소는 순서대로 접근
            self.listWidget.addItem(display_text)
    
    #수정하기 모드 메서드        
    def toggle_fix_mode(self):
                #'수정' 모드일 때, 두 그룹 모두 활성화합니다.
        self.enable_group(self.readonly_group)
        self.enable_group(self.editable_group)
        
        self.current_mode = 'fix'
                #수정 모드일 때, 선택한 데이터를 기본값으로 설정합니다.
        self.fix_barcode.setText(self.chose_barcode.text())
        self.fix_proname.setText(self.chose_proname.text())
        self.fix_price.setText(self.chose_price.text())
               
                #그룹 초기화합니다.
        self.clear_group(self.readonly_group)
        self.clear_group(self.editable_group)

    #추가하기 모드 메서드
    def toggle_add_mode(self):
                #'추가' 모드일 때, 수정할 그룹은 비활성화하고, 읽기 전용 그룹은 활성화합니다.
        self.disable_group(self.readonly_group)
        self.enable_group(self.editable_group)
        
        self.current_mode = 'add'
                # 그룹 초기화합니다.
        self.clear_group(self.readonly_group)
        self.clear_group(self.editable_group)
    
    #삭제하기 모드 메서드
    def toggle_delete_mode(self):
                #'삭제' 모드일 때, 수정할 그룹은 비활성화하고, 읽기 전용 그룹은 활성화합니다.
        self.disable_group(self.editable_group)
        self.enable_group(self.readonly_group)
        self.current_mode = 'delete'
                # 그룹 초기화합니다.
        self.clear_group(self.readonly_group)
        self.clear_group(self.editable_group)
    
    #listWidget 항목 선택 메서드        
    def on_item_clicked(self, item):
                    # 선택된 항목의 텍스트를 \t 기준으로 파싱
        selected_text = item.text()
        parts = selected_text.split('\t')
        
        self.chose_barcode.setText(parts[0])               #파싱데이터 - 바코드
        self.chose_proname.setText(parts[2])               #파싱데이터 - 제품명
        self.chose_price.setText(parts[1].split('원')[0])  #파싱데이터 - 가격
    
    
    #확인 버튼 메서드
    def on_pushbutton_ok_clicked(self):
            #수정하기 모드
        if self.current_mode == 'fix':
            barcode = self.fix_barcode.text()
            proname = self.fix_proname.text()
            price = self.fix_price.text()
            chose_barcode = self.chose_barcode.text()
            
                    # 입력된 데이터가 없을 경우, 선택된 데이터로 대체합니다.
            if not barcode:
                barcode = chose_barcode
            if not proname:
                proname = self.chose_proname.text()
            if not price:
                price = self.chose_price.text()
                        
            Database.product_fix(chose_barcode, barcode, proname, price) #수정하는 db 쿼리문 실행
           
           #추가하기 모드 
        elif self.current_mode == 'add':
            barcode = self.fix_barcode.text()
            proname = self.fix_proname.text()
            price = self.fix_price.text()
            
                # 입력 데이터 자료형 검사
            try:
                barcode = int(barcode)  # 바코드는 정수형이어야 함
                price = int(price)      # 가격은 정수형이어야 함
            except ValueError:
                QMessageBox.critical(self, '입력 오류', '바코드와 가격은 숫자로 입력해 주세요.')
                return        
            
                # 입력 데이터가 없을 경우
            if not barcode or not proname or not price:
                QMessageBox.critical(self, '입력 오류', '바코드와 가격은 숫자로 입력해 주세요.')
                # 입력 데이터가 있을 경우 - 정상 상황
            else:
                check1, check2, check3 = Database.product_find(barcode)
                    
                if check2 == 'none':    #정상 상황
                    Database.product_add(barcode, proname, price)
                else:                   #바코드 정보가 이미 존재할때
                    QMessageBox.critical(self, '입력 오류', '바코드 정보가 이미 존재합니다.')
            
            #삭제하기 모드
        elif self.current_mode == 'delete':
            
            chose_barcode = self.chose_barcode.text()
            
            if not chose_barcode:
                QMessageBox.critical(self, '입력 오류', '데이터를 선택해 주세요')
            else:
                Database.product_delete(chose_barcode)
            
        
        self.listWidget.clear()  # 기존 리스트 초기화
        self.load_data()  # 새로 데이터 로드
        
                        # 그룹 초기화합니다.
        self.clear_group(self.readonly_group)
        self.clear_group(self.editable_group)



    #그룹 활성화 메서드    
    def enable_group(self, group):
        for widget in group:
            widget.setEnabled(True)
    #그룹 비활성화 메서드
    def disable_group(self, group):
        for widget in group:
            widget.setEnabled(False)     
    #그룹 초기화 메서드
    def clear_group(self, group):
        for widget in group:
            widget.setText('')
    
    def go_to_admin_mode(self): #뒤로가기 버튼
        self.close()  # 두 번째 화면 닫기
        self.third_window = ThirdWindow(self.mqtt_client)
        self.third_window.show() 
                    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
