import string

def phoneb(n, pb) :
    pb = {'Name': 'Number'}
    i = 0
    for i in range(n) :
        st = input()
        st.split(" ")
        print(st)
        pb[st[0]] = st[1]
        
if __name__ == '__main__' :
    n = int(input())
    pb = {}
    phoneb(n, pb)
    print(pb)
