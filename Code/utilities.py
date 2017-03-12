import re

def generateCandidates(w1, w2):
    n=len(w1)
    m=len(w2)
    all_candidates = []
    for i in range(1,n+1):
        for j in range(m):
            all_candidates.append(w1[0:i]+w2[j:])
    #print "len(all_candidates)= ",len(all_candidates)
    all_candidates=set(all_candidates)
    #print "len(all_candidates) after deduplication= ",len(all_candidates)
    return all_candidates

if __name__=="__main__":
    print generateCandidates("abc","cdef")
