import cv2

while(True):
    Img = cv2.imread ("dataset\Phat.1910429.3.jpg",1)
    cv2.imshow('frame',Img) 
    key=cv2.waitKey(1) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# # import sqlite3
# # from datetime import datetime
# # import time

# # conn=sqlite3.connect('mydb.db')
# # cursor=conn.execute("SELECT MSSV FROM people")
# # prev_num_rows=cursor.fetchall()
# # print(prev_num_rows[0][0])
# # print(type(prev_num_rows))
# # print(len(prev_num_rows))
# # l1=[(123,),(456,),(789,)]
# # l2=[(456,),(123,)]
# # dif=list(set(l1)^set(l2))
# # print(dif)
# # # profile=[]
# # # for row in cursor:
# # #     profile.append(row)
# # # #print(prev_num_rows)
# # # #print(type(prev_num_rows))
# # # print(profile)
# # # print(type(profile))
# # # print(type(profile[1]))
# from PIL import Image

# def truncate(value):
#     if (value < 0):
#         return 0
#     if (value > 255):
#         return 255
#     return value

# if __name__ == "__main__":
#     img = Image.open('dataset/Phat.1910429.2.jpg')
#     pixels = img.load()
#     print(type(pixels))

#     img_new = Image.new(img.mode, img.size)
#     pixels_new = img_new.load()qq
#     brightness = 20
#     for i in range(img_new.size[0]):
#         for j in range(img_new.size[1]):
#             r, b, g = pixels[i,j]
#             _r = truncate(r + brightness)
#             _b = truncate(b + brightness)
#             _g = truncate(g + brightness)
#             pixels_new[i,j] = (_r, _b, _g, 255)
#     img_new.show()

