

class Warping:

    def sdtw(x, y, w=float('inf')):
        """
        subsequence dynamic time warping.
        """
        N = len(x)
        M = len(y)
        dtw = [[float('inf')] * (M+1)] * (N+1)

        w = max(w, abs(N-M))

        dtw[0][0] = 0

        if w > abs(N-M):
            for i in range(1, N+1):
                for j in range(max(1, i-w), min(M+1, i+w)):
                    dtw[i][j] = 0

        for i in range(1, N+1):
            for j in range(max(1, i-w), min(M+1, i+w)):
                cost = abs(x[i-1]-y[j-1])
                dtw[i][j] = cost + min(dtw[i-1][j],
                                       dtw[i][j-1],
                                       dtw[i-1][j-1])
        
        return dtw[N][M]
    
    def mtmm_dtw():
        """
        Multi-template multi-match dynamic time warping.
        """
        pass

        
