from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
import sys
from matplotlib import pyplot as plt
import numpy as np

# นำเข้า namespaces
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import FaceDetectionModel, FaceRecognitionModel, FaceAttributeTypeDetection01
from azure.core.credentials import AzureKeyCredential



def main():
    global face_client

    try:
        # รับค่าการตั้งค่าการกำหนดค่า
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
        ai_key = os.getenv('AI_SERVICE_KEY')

        # ตรวจสอบการตั้งค่า
        print("=== การตรวจสอบการตั้งค่า ===")
        print(f"AI Endpoint: {ai_endpoint}")
        print(f"AI Key: {'***' + ai_key[-4:] if ai_key else 'None'}")
        
        if not ai_endpoint or not ai_key:
            print("ข้อผิดพลาด: ไม่พบ AI_SERVICE_ENDPOINT หรือ AI_SERVICE_KEY")
            print("กรุณาสร้างไฟล์ .env และเพิ่ม:")
            print("AI_SERVICE_ENDPOINT=your_endpoint_here")
            print("AI_SERVICE_KEY=your_key_here")
            return

        # รับรูปภาพ
        image_file = 'images/people.jpg'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        print(f"กำลังอ่านไฟล์รูปภาพ: {image_file}")
        
        # ตรวจสอบว่าไฟล์รูปภาพมีอยู่จริง
        if not os.path.exists(image_file):
            print(f"ข้อผิดพลาด: ไม่พบไฟล์รูปภาพ {image_file}")
            print("กรุณาตรวจสอบว่าไฟล์มีอยู่จริงหรือระบุพาธที่ถูกต้อง")
            return

        with open(image_file, "rb") as f:
            image_data = f.read()

        print(f"อ่านไฟล์รูปภาพสำเร็จ: {len(image_data)} bytes")

        # ตรวจสอบสิทธิ์ Face client
        print("กำลังเชื่อมต่อกับ Azure Face API...")
        face_client = FaceClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key))
        
        print("เชื่อมต่อสำเร็จ!")
        
        # วิเคราะห์รูปภาพ
        AnalyzeImage(image_file, image_data, face_client)

    except FileNotFoundError as e:
        print(f"ไม่พบไฟล์: {e}")
    except Exception as ex:
        print(f"ข้อผิดพลาด: {ex}")
        import traceback
        traceback.print_exc()


def AnalyzeImage(filename, image_data, face_client):
    print('\nกำลังวิเคราะห์ ', filename)

    # ตรวจจับใบหน้าด้วย Face API พร้อมคุณลักษณะเพิ่มเติม
    detected_faces = face_client.detect(
        image_content=image_data,
        detection_model=FaceDetectionModel.DETECTION01,
        recognition_model=FaceRecognitionModel.RECOGNITION01,
        return_face_id=False,
        return_face_attributes=[
            FaceAttributeTypeDetection01.HEAD_POSE,
            FaceAttributeTypeDetection01.OCCLUSION,
            FaceAttributeTypeDetection01.ACCESSORIES
        ]
    )

    # ระบุผู้คนในรูปภาพ
    if detected_faces and len(detected_faces) > 0:
        print(f"\nตรวจพบใบหน้า {len(detected_faces)} ใบหน้า:")

        # เตรียมรูปภาพสำหรับการวาด
        image = Image.open(filename)
        fig = plt.figure(figsize=(image.width/100, image.height/100))
        plt.axis('off')
        draw = ImageDraw.Draw(image)
        
        # สีที่หลากหลายสำหรับใบหน้าแต่ละใบ
        colors = ['cyan', 'red', 'green', 'yellow', 'magenta', 'orange']

        # วาดกล่องขอบเขตรอบผู้คนที่ตรวจพบ
        face_count = 0
        for face in detected_faces:
            face_count += 1
            color = colors[face_count % len(colors)]
            
            # รับพิกัดสี่เหลี่ยมใบหน้า
            left = face.face_rectangle.left
            top = face.face_rectangle.top
            width = face.face_rectangle.width
            height = face.face_rectangle.height
            
            # วาดสี่เหลี่ยม
            bounding_box = [(left, top), (left + width, top + height)]
            draw.rectangle(bounding_box, outline=color, width=3)
            
            # เพิ่มหมายเลขใบหน้า
            draw.text((left, top-20), f"Face {face_count}", fill=color)
            
            # แสดงรายละเอียดใบหน้า
            print(f"\n--- ใบหน้าที่ {face_count} ---")
            print(f"ตำแหน่ง: ({left}, {top})")
            print(f"ขนาด: {width} x {height}")
            
            # แสดงคุณลักษณะหากมี
            if hasattr(face, 'face_attributes') and face.face_attributes:
                if face.face_attributes.head_pose:
                    print(f"ท่าหัว - Yaw: {face.face_attributes.head_pose.yaw:.1f}°")
                    print(f"ท่าหัว - Pitch: {face.face_attributes.head_pose.pitch:.1f}°")
                    print(f"ท่าหัว - Roll: {face.face_attributes.head_pose.roll:.1f}°")
                
                if face.face_attributes.occlusion:
                    print(f"หน้าผากถูกบัง: {face.face_attributes.occlusion.forehead_occluded}")
                    print(f"ตาถูกบัง: {face.face_attributes.occlusion.eye_occluded}")
                    print(f"ปากถูกบัง: {face.face_attributes.occlusion.mouth_occluded}")
                
                if face.face_attributes.accessories:
                    accessories = [acc.type for acc in face.face_attributes.accessories]
                    print(f"อุปกรณ์เสริม: {', '.join(accessories) if accessories else 'ไม่มี'}")

        # สร้างสถิติสรุป
        print(f"\n=== สรุปผลการตรวจจับ ===")
        print(f"จำนวนใบหน้าทั้งหมด: {len(detected_faces)}")
        print(f"ไฟล์รูปภาพ: {filename}")
        print(f"ขนาดรูปภาพ: {image.width} x {image.height} พิกเซล")
            
        # บันทึกรูปภาพที่มีการใส่หมายเหตุ
        plt.imshow(image)
        plt.tight_layout(pad=0)
        outputfile = f'detected_faces_{len(detected_faces)}_faces.jpg'
        fig.savefig(outputfile, dpi=150, bbox_inches='tight')
        print(f'ผลลัพธ์ถูกบันทึกใน {outputfile}')
        
        # สร้างรายงาน
        create_face_report(detected_faces, filename)
        
    else:
        print("\nไม่พบใบหน้าในรูปภาพ")

def create_face_report(faces, filename):
    """สร้างรายงานผลการตรวจจับใบหน้า"""
    import datetime
    
    report_file = f'face_report_{len(faces)}_faces.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== รายงานการตรวจจับใบหน้า ===\n")
        f.write(f"ไฟล์: {filename}\n")
        f.write(f"จำนวนใบหน้า: {len(faces)}\n")
        f.write(f"วันที่วิเคราะห์: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, face in enumerate(faces, 1):
            f.write(f"ใบหน้าที่ {i}:\n")
            f.write(f"  ตำแหน่ง: ({face.face_rectangle.left}, {face.face_rectangle.top})\n")
            f.write(f"  ขนาด: {face.face_rectangle.width} x {face.face_rectangle.height}\n")
            if hasattr(face, 'face_attributes') and face.face_attributes:
                if face.face_attributes.head_pose:
                    f.write(f"  ท่าหัว: Yaw={face.face_attributes.head_pose.yaw:.1f}°, Pitch={face.face_attributes.head_pose.pitch:.1f}°, Roll={face.face_attributes.head_pose.roll:.1f}°\n")
                if face.face_attributes.occlusion:
                    f.write(f"  การบัง: หน้าผาก={face.face_attributes.occlusion.forehead_occluded}, ตา={face.face_attributes.occlusion.eye_occluded}, ปาก={face.face_attributes.occlusion.mouth_occluded}\n")
                if face.face_attributes.accessories:
                    accessories = [str(acc.type) for acc in face.face_attributes.accessories]
                    f.write(f"  อุปกรณ์เสริม: {', '.join(accessories) if accessories else 'ไม่มี'}\n")
            f.write("\n")
    
    print(f'รายงานถูกบันทึกใน {report_file}')

if __name__ == "__main__":
    print("เริ่มต้นโปรแกรมตรวจจับใบหน้า")
    main()
    print("โปรแกรมทำงานเสร็จสิ้น")