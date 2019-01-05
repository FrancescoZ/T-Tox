#!/usr/bin/env python
while True:
    print("Welcome to T-Tox, there are a few different options to use this program:")
    print(" 1- Pretrained Model")
    print(" 1- Personalized pretrained Model")
    print(" 2- Train new Model with standard data")
    print(" 3- Train new Model with personalized data")
    print(" 4- Use package")
    print(" 5- Exit")
    print("All these modes can be called using the respectfully python code. This is only an interface to help the user navigate into the different files.")
    mode = input("Usage mode: ")
    #Pretrained model
    if mode == "1":
        mode = 1
        break
    elif mode == "2":
        mode = 2
        break
    elif mode == "3":
        mode = 2
        break
    elif mode == "4":
        print("Unfortunatly this interface was not created to be an interactive documentation, however you can refer to the pdf at this path: TODOPATH")
        break
    elif mode == "5":
        break    
    else:
        print("Invalid choice")
print("Thank you for using T-Tox. Hope to see you soon")
exit()