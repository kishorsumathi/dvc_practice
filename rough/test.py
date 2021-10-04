
import os

def main():
    print("inside main")
    with open(os.path.join("rough","test.txt"),"w") as f:
        f.write("this is a test file")
if __name__=="__main__":
    main()